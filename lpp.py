#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LIGGGHTS Post Processing (Python 3)
-------------------------------------------

This single-file tool merges the original Python2-era `lpp.py`, `vtk.py`, and `dump.py`
into one Python 3 script, adds binary/ASCII VTK output selection, and avoids the
name-collision with the third‑party `vtk` package by providing two backends:

  1) Legacy writer (default, no external deps): writes VTK Legacy .vtk files
     in ASCII or BINARY (big-endian) format.

  2) vtk library backend (optional, requires `pip install vtk`): if selected, uses
     VTK’s Python bindings to write equivalent .vtk files. This is enabled via
     `--backend vtk`. If the library is not available, it automatically falls back
     to the legacy writer.

Command-line usage (compatible with original options + new ones):

  python lpp.py [options] dump.file [dump2 ...]

Important options (original + new):

  -o, --output ROOT           Output file name/prefix (default: liggghts + timestep)
  --chunksize N               Files per chunk (default: 8)
  --cpunum N                  Number of worker processes (default: CPU cores)
  --no-overwrite              Do not re-generate existing .vtk files
  --debug                     Verbose debug output
  --quiet                     Minimal output (overrides --debug)
  --format {ascii,binary}     VTK legacy output format (default: ascii)
  --backend {legacy,vtk}      Writer backend. Default 'legacy'. 'vtk' uses python-vtk.
  --help                      Show help

Notes
-----
* The reading/wrangling functionality is ported from Pizza.py-style tools and
  implements snapshot selection, mapping, scaling, etc.
* `--format binary` writes big‑endian legacy VTK as per VTK legacy spec.
* Surface triangle output is supported in both ASCII and binary.
* Bounding-box files (RECTILINEAR_GRID) are written alongside particle files.
* If you select `--backend vtk` and the `vtk` package is installed, the writer will
  use `vtkPolyDataWriter`/`vtkRectilinearGridWriter` and respect ASCII/BINARY setting.

Tested with Python 3.8+ and NumPy.

License
-------
Retains the original GNU GPL notice from Pizza.py components.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import glob
import multiprocessing as mp
import os
import re
import struct
import sys
from math import floor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional vtk backend (import lazily only if requested)
_VTK_AVAILABLE = False
_vtk = None

PIZZA_GUNZIP = os.environ.get("PIZZA_GUNZIP", "gunzip")

# -----------------------------------------------------------------------------
# Utility: robust prints depending on quiet/debug flags
# -----------------------------------------------------------------------------

def _println(*args, file=sys.stdout, quiet=False, end="\n"):
    if not quiet:
        print(*args, file=file, end=end)
        file.flush()

# -----------------------------------------------------------------------------
# Dump reader (port of dump.py with Python 3 fixes)
# -----------------------------------------------------------------------------

class Snap:
    """One LAMMPS dump snapshot."""
    __slots__ = (
        "time",
        "tselect",
        "natoms",
        "nselect",
        "aselect",
        "xlo", "xhi", "ylo", "yhi", "zlo", "zhi",
        "atoms",
    )
    def __init__(self):
        self.time: int = 0
        self.tselect: int = 0
        self.natoms: int = 0
        self.nselect: int = 0
        self.aselect: np.ndarray = np.zeros(0, dtype=int)
        self.xlo = self.xhi = self.ylo = self.yhi = self.zlo = self.zhi = 0.0
        self.atoms: Optional[np.ndarray] = None


class dump:
    """
    Read, write, manipulate dump files and particle attributes (Python 3 port).
    This class closely follows the original Pizza.py 'dump' tool.
    """

    # ---------------------------------------------------------------------

    def __init__(self, *input, **kwargs):
        self.snaps: List[Snap] = []
        self.nsnaps = self.nselect = 0
        self.names: Dict[str, int] = {}
        self.tselect = tselect(self)
        self.aselect = aselect(self)
        self.atype = "type"
        self.bondflag = 0
        self.bondlist = []
        self.triflag = 0
        self.trilist = []
        self.lineflag = 0
        self.linelist = []
        self.multiprocflag = 0
        self.fileNums: List[int] = []
        self.objextra = None

        outputfl = kwargs.get("output", True)

        if isinstance(input[0], dict):
            # multiprocessing path
            dictionary = input[0]
            outputfl = dictionary.get("debugMode", False)
            _println(f"subprocess pid: {os.getpid()}", quiet=not outputfl)
            self.flist = dictionary["filelist"]
            self.multiprocflag = 1
            self.increment = 0
            self.read_all(output=outputfl)
        else:
            # serial path
            words = input[0].split()
            self.flist = []
            for word in words:
                self.flist += glob.glob(word)
            if len(self.flist) == 0 and len(input) == 1:
                raise Exception("no dump file specified")

            if len(input) == 1:
                self.increment = 0
                self.read_all(output=outputfl)
            else:
                self.increment = 1
                self.nextfile = 0
                self.eof = 0

    # ---------------------------------------------------------------------

    def read_all(self, **kwargs):
        outputfl: bool = kwargs.get("output", True)
        _println("reading dump file...", quiet=not outputfl)

        for file in self.flist:
            if file.endswith(".gz"):
                f = gzip.open(file, "rt")
            else:
                f = open(file, "r")

            try:
                snap = self.read_snapshot(f)
                while snap:
                    self.snaps.append(snap)
                    _println(snap.time, end="", quiet=not outputfl)
                    self.fileNums.append(snap.time)
                    snap = self.read_snapshot(f)
            finally:
                f.close()

        _println("", quiet=not outputfl)

        # sort entries by timestep, cull duplicates
        self.snaps.sort(key=functools.cmp_to_key(self.compare_time))
        self.fileNums.sort()
        self.cull()
        self.nsnaps = len(self.snaps)

        # select all timesteps and atoms
        self.tselect.all(output=outputfl)

        # announce column names
        if len(self.snaps) == 0:
            _println("no column assignments made", quiet=not outputfl)
        elif len(self.names):
            _println("assigned columns: " + self.names2str(), quiet=not outputfl)
        else:
            _println("no column assignments made", quiet=not outputfl)

        # if snapshots are scaled, unscale them
        if ("x" not in self.names) or ("y" not in self.names) or ("z" not in self.names):
            _println("dump scaling status is unknown", quiet=not outputfl)
        elif self.nsnaps > 0:
            if self.scale_original == 1:
                self.unscale()
            elif self.scale_original == 0:
                _println("dump is already unscaled", quiet=not outputfl)
            else:
                _println("dump scaling status is unknown", quiet=not outputfl)

    # ---------------------------------------------------------------------

    def next(self):
        if not getattr(self, "increment", 0):
            raise Exception("cannot read incrementally")

        while True:
            f = open(self.flist[self.nextfile], "rb")
            f.seek(self.eof)
            snap = self.read_snapshot(f)
            if not snap:
                self.nextfile += 1
                if self.nextfile == len(self.flist):
                    return -1
                f.close()
                self.eof = 0
                continue
            self.eof = f.tell()
            f.close()
            try:
                self.findtime(snap.time)
                continue
            except Exception:
                break

        self.snaps.append(snap)
        snap = self.snaps[self.nsnaps]
        snap.tselect = 1
        snap.nselect = snap.natoms
        for i in range(snap.natoms):
            snap.aselect[i] = 1
        self.nsnaps += 1
        self.nselect += 1
        return snap.time

    # ---------------------------------------------------------------------

    def read_snapshot(self, f) -> Optional[Snap]:
        """
        Read a single snapshot from a LAMMPS-style dump file handle.
        """
        try:
            snap = Snap()
            item = f.readline()            # "ITEM: TIMESTEP"
            if not item:
                return None
            time_line = f.readline()       # timestep value
            if not time_line:
                return None
            snap.time = int(time_line.split()[0])

            item = f.readline()            # "ITEM: NUMBER OF ATOMS"
            if not item:
                return None
            natoms_line = f.readline()
            if not natoms_line:
                return None
            snap.natoms = int(natoms_line)

            snap.aselect = np.zeros(snap.natoms, dtype=int)

            item = f.readline()            # "ITEM: BOX BOUNDS ..."
            words = f.readline().split()
            snap.xlo, snap.xhi = float(words[0]), float(words[1])
            words = f.readline().split()
            snap.ylo, snap.yhi = float(words[0]), float(words[1])
            words = f.readline().split()
            snap.zlo, snap.zhi = float(words[0]), float(words[1])

            item = f.readline()            # "ITEM: ATOMS ..."
            if len(self.names) == 0:
                self.scale_original = -1
                xflag = yflag = zflag = -1
                words = item.split()[2:]
                if len(words):
                    for i, w in enumerate(words):
                        if w in ("x", "xu"):
                            xflag = 0
                            self.names["x"] = i
                        elif w in ("xs", "xsu"):
                            xflag = 1
                            self.names["x"] = i
                        elif w in ("y", "yu"):
                            yflag = 0
                            self.names["y"] = i
                        elif w in ("ys", "ysu"):
                            yflag = 1
                            self.names["y"] = i
                        elif w in ("z", "zu"):
                            zflag = 0
                            self.names["z"] = i
                        elif w in ("zs", "zsu"):
                            zflag = 1
                            self.names["z"] = i
                        else:
                            self.names[w] = i
                    if xflag == 0 and yflag == 0 and zflag == 0:
                        self.scale_original = 0
                    if xflag == 1 and yflag == 1 and zflag == 1:
                        self.scale_original = 1

            if snap.natoms:
                words = f.readline().split()
                ncol = len(words)
                for _ in range(1, snap.natoms):
                    words += f.readline().split()
                floats = list(map(float, words))
                atoms = np.zeros((snap.natoms, ncol), dtype=float)
                start = 0
                stop = ncol
                for i in range(snap.natoms):
                    atoms[i] = floats[start:stop]
                    start = stop
                    stop += ncol
            else:
                atoms = None

            snap.atoms = atoms
            return snap
        except Exception:
            return None

    # ---------------------------------------------------------------------

    def map(self, *pairs):
        if len(pairs) % 2 != 0:
            raise Exception("dump map() requires pairs of mappings")
        for i in range(0, len(pairs), 2):
            j = i + 1
            self.names[pairs[j]] = pairs[i] - 1

    # ---------------------------------------------------------------------

    def delete(self):
        ndel = i = 0
        while i < self.nsnaps:
            if not self.snaps[i].tselect:
                del self.fileNums[i]
                del self.snaps[i]
                self.nsnaps -= 1
                ndel += 1
            else:
                i += 1
        _println(f"{ndel} snapshots deleted", quiet=False)
        _println(f"{self.nsnaps} snapshots remaining", quiet=False)

    # ---------------------------------------------------------------------

    def scale(self, *lst):
        x = self.names["x"]
        y = self.names["y"]
        z = self.names["z"]
        if len(lst) == 0:
            _println("Scaling dump ...")
            for snap in self.snaps:
                self.scale_one(snap, x, y, z)
        else:
            i = self.findtime(lst[0])
            self.scale_one(self.snaps[i], x, y, z)

    def scale_one(self, snap: Snap, x, y, z):
        xprdinv = 1.0 / (snap.xhi - snap.xlo)
        yprdinv = 1.0 / (snap.yhi - snap.ylo)
        zprdinv = 1.0 / (snap.zhi - snap.zlo)
        atoms = snap.atoms
        atoms[:, x] = (atoms[:, x] - snap.xlo) * xprdinv
        atoms[:, y] = (atoms[:, y] - snap.ylo) * yprdinv
        atoms[:, z] = (atoms[:, z] - snap.zlo) * zprdinv

    # ---------------------------------------------------------------------

    def unscale(self, *lst):
        x = self.names["x"]
        y = self.names["y"]
        z = self.names["z"]
        if len(lst) == 0:
            _println("Unscaling dump ...")
            for snap in self.snaps:
                self.unscale_one(snap, x, y, z)
        else:
            i = self.findtime(lst[0])
            self.unscale_one(self.snaps[i], x, y, z)

    def unscale_one(self, snap: Snap, x, y, z):
        xprd = snap.xhi - snap.xlo
        yprd = snap.yhi - snap.ylo
        zprd = snap.zhi - snap.zlo
        atoms = snap.atoms
        atoms[:, x] = snap.xlo + atoms[:, x] * xprd
        atoms[:, y] = snap.ylo + atoms[:, y] * yprd
        atoms[:, z] = snap.zlo + atoms[:, z] * zprd

    # ---------------------------------------------------------------------

    def wrap(self):
        _println("Wrapping dump ...")
        x = self.names["x"]; y = self.names["y"]; z = self.names["z"]
        ix = self.names["ix"]; iy = self.names["iy"]; iz = self.names["iz"]
        for snap in self.snaps:
            xprd = snap.xhi - snap.xlo
            yprd = snap.yhi - snap.ylo
            zprd = snap.zhi - snap.zlo
            atoms = snap.atoms
            atoms[:, x] -= atoms[:, ix] * xprd
            atoms[:, y] -= atoms[:, iy] * yprd
            atoms[:, z] -= atoms[:, iz] * zprd

    def unwrap(self):
        _println("Unwrapping dump ...")
        x = self.names["x"]; y = self.names["y"]; z = self.names["z"]
        ix = self.names["ix"]; iy = self.names["iy"]; iz = self.names["iz"]
        for snap in self.snaps:
            xprd = snap.xhi - snap.xlo
            yprd = snap.yhi - snap.ylo
            zprd = snap.zhi - snap.zlo
            atoms = snap.atoms
            atoms[:, x] += atoms[:, ix] * xprd
            atoms[:, y] += atoms[:, iy] * yprd
            atoms[:, z] += atoms[:, iz] * zprd

    # ---------------------------------------------------------------------

    def owrap(self, other: str):
        _println("Wrapping to other ...")
        idc = self.names["id"]
        x = self.names["x"]; y = self.names["y"]; z = self.names["z"]
        ix = self.names["ix"]; iy = self.names["iy"]; iz = self.names["iz"]
        iother = self.names[other]

        for snap in self.snaps:
            xprd = snap.xhi - snap.xlo
            yprd = snap.yhi - snap.ylo
            zprd = snap.zhi - snap.zlo
            atoms = snap.atoms
            ids = {atoms[i][idc]: i for i in range(snap.natoms)}
            for i in range(snap.natoms):
                j = ids[atoms[i][iother]]
                atoms[i][x] += (atoms[i][ix] - atoms[j][ix]) * xprd
                atoms[i][y] += (atoms[i][iy] - atoms[j][iy]) * yprd
                atoms[i][z] += (atoms[i][iz] - atoms[j][iz]) * zprd
            if self.lineflag == 2 or self.triflag == 2:
                self.objextra.owrap(
                    snap.time, xprd, yprd, zprd, ids, atoms, iother, ix, iy, iz
                )

    # ---------------------------------------------------------------------

    def names2str(self) -> str:
        if not self.names:
            return ""
        max_idx = max(self.names.values())
        # reverse mapping index -> name
        rev = {idx: name for name, idx in self.names.items()}
        ordered = [rev[i] for i in range(max_idx + 1) if i in rev]
        return " ".join(ordered)

    # ---------------------------------------------------------------------

    def sort(self, *lst, **kwargs):
        outputfl = kwargs.get("output", True)
        if len(lst) == 0:
            _println("Sorting selected snapshots ...", quiet=not outputfl)
            idc = self.names["id"]
            for snap in self.snaps:
                if snap.tselect:
                    self.sort_one(snap, idc)
        elif isinstance(lst[0], str):
            _println(f"Sorting selected snapshots by {lst[0]} ...", quiet=not outputfl)
            idc = self.names[lst[0]]
            for snap in self.snaps:
                if snap.tselect:
                    self.sort_one(snap, idc)
        else:
            i = self.findtime(lst[0])
            idc = self.names["id"]
            self.sort_one(self.snaps[i], idc)

    def sort_one(self, snap: Snap, idc):
        atoms = snap.atoms
        ids = atoms[:, idc]
        ordering = np.argsort(ids)
        for i in range(atoms.shape[1]):
            atoms[:, i] = np.take(atoms[:, i], ordering)

    # ---------------------------------------------------------------------

    def write(self, file, header=1, append=0):
        namestr = self.names2str() if len(self.snaps) else ""
        mode = "a" if append else "w"
        with open(file, mode) as f:
            for snap in self.snaps:
                if not snap.tselect:
                    continue
                _println(snap.time, end=" ")
                if header:
                    print("ITEM: TIMESTEP", file=f)
                    print(snap.time, file=f)
                    print("ITEM: NUMBER OF ATOMS", file=f)
                    print(snap.nselect, file=f)
                    print("ITEM: BOX BOUNDS", file=f)
                    print(snap.xlo, snap.xhi, file=f)
                    print(snap.ylo, snap.yhi, file=f)
                    print(snap.zlo, snap.zhi, file=f)
                    print("ITEM: ATOMS", namestr, file=f)
                atoms = snap.atoms
                nvalues = atoms.shape[1]
                for i in range(snap.natoms):
                    if not snap.aselect[i]:
                        continue
                    parts = []
                    for j in range(nvalues):
                        if j < 2:
                            parts.append(str(int(atoms[i][j])))
                        else:
                            parts.append(str(atoms[i][j]))
                    print(" ".join(parts), file=f)
        _println(f"\n{self.nselect} snapshots")

    def scatter(self, root):
        namestr = self.names2str() if len(self.snaps) else ""
        for snap in self.snaps:
            if not snap.tselect:
                continue
            _println(snap.time, end=" ")
            file = f"{root}.{snap.time}"
            with open(file, "w") as f:
                print("ITEM: TIMESTEP", file=f)
                print(snap.time, file=f)
                print("ITEM: NUMBER OF ATOMS", file=f)
                print(snap.nselect, file=f)
                print("ITEM: BOX BOUNDS", file=f)
                print(snap.xlo, snap.xhi, file=f)
                print(snap.ylo, snap.yhi, file=f)
                print(snap.zlo, snap.zhi, file=f)
                print("ITEM: ATOMS", namestr, file=f)
                atoms = snap.atoms
                nvalues = atoms.shape[1]
                for i in range(snap.natoms):
                    if not snap.aselect[i]:
                        continue
                    parts = []
                    for j in range(nvalues):
                        if j < 2:
                            parts.append(str(int(atoms[i][j])))
                        else:
                            parts.append(str(atoms[i][j]))
                    print(" ".join(parts), file=f)
        _println(f"\n{self.nselect} snapshots")

    # ---------------------------------------------------------------------

    def minmax(self, colname: str) -> Tuple[float, float]:
        icol = self.names[colname]
        minv = 1.0e20
        maxv = -minv
        for snap in self.snaps:
            if not snap.tselect:
                continue
            atoms = snap.atoms
            for i in range(snap.natoms):
                if not snap.aselect[i]:
                    continue
                if atoms[i][icol] < minv:
                    minv = atoms[i][icol]
                if atoms[i][icol] > maxv:
                    maxv = atoms[i][icol]
        return (minv, maxv)

    def set(self, eq: str):
        """
        Set a column via an equation: e.g. d.set("$ke = $vx*$vx + $vy*$vy")
        """
        _println("Setting ...")
        pattern = r"\$\w*"
        found = re.findall(pattern, eq)

        lhs = found[0][1:]
        if lhs not in self.names:
            self.newcolumn(lhs)

        for item in found:
            name = item[1:]
            column = self.names[name]
            insert = f"snap.atoms[i][{column}]"
            eq = eq.replace(item, insert)

        ceq = compile(eq, "", "exec")

        for snap in self.snaps:
            if not snap.tselect:
                continue
            for i in range(snap.natoms):
                if snap.aselect[i]:
                    local_env = {"snap": snap, "i": i}
                    exec(ceq, {}, local_env)

    def setv(self, colname: str, vec: Iterable[float]):
        _println("Setting ...")
        if colname not in self.names:
            self.newcolumn(colname)
        icol = self.names[colname]
        for snap in self.snaps:
            if not snap.tselect:
                continue
            if snap.nselect != len(vec):
                raise Exception("vec length does not match # of selected atoms")
            atoms = snap.atoms
            m = 0
            for i in range(snap.natoms):
                if snap.aselect[i]:
                    atoms[i][icol] = vec[m]
                    m += 1

    def clone(self, nstep: int, col: str):
        istep = self.findtime(nstep)
        icol = self.names[col]
        idc = self.names["id"]
        ids = {self.snaps[istep].atoms[i][idc]: i for i in range(self.snaps[istep].natoms)}
        for snap in self.snaps:
            if not snap.tselect:
                continue
            atoms = snap.atoms
            for i in range(snap.natoms):
                if not snap.aselect[i]:
                    continue
                j = ids[atoms[i][idc]]
                atoms[i][icol] = self.snaps[istep].atoms[j][icol]

    def spread(self, old: str, n: int, new: str):
        iold = self.names[old]
        if new not in self.names:
            self.newcolumn(new)
        inew = self.names[new]

        minv, maxv = self.minmax(old)
        _println(f"min/max = {minv} {maxv}")
        gap = maxv - minv
        invdelta = n / gap if gap != 0 else 0.0
        for snap in self.snaps:
            if not snap.tselect:
                continue
            atoms = snap.atoms
            for i in range(snap.natoms):
                if not snap.aselect[i]:
                    continue
                ivalue = int((atoms[i][iold] - minv) * invdelta) + 1
                if ivalue > n:
                    ivalue = n
                if ivalue < 1:
                    ivalue = 1
                atoms[i][inew] = ivalue

    def time(self) -> List[int]:
        return [snap.time for snap in self.snaps if snap.tselect]

    def atom(self, n: int, *lst):
        if len(lst) == 0:
            raise Exception("no columns specified")
        columns = [self.names[name] for name in lst]
        values = [[0] * self.nselect for _ in columns]
        idc = self.names["id"]
        m = 0
        for snap in self.snaps:
            if not snap.tselect:
                continue
            atoms = snap.atoms
            i = None
            for k in range(snap.natoms):
                if int(atoms[k][idc]) == n:
                    i = k
                    break
            if i is None or int(atoms[i][idc]) != n:
                raise Exception("could not find atom ID in snapshot")
            for j, col in enumerate(columns):
                values[j][m] = atoms[i][col]
            m += 1
        return values[0] if len(lst) == 1 else values

    def vecs(self, n: int, *lst):
        snap = self.snaps[self.findtime(n)]
        if len(lst) == 0:
            raise Exception("no columns specified")
        columns = [self.names[name] for name in lst]
        values = [[0] * snap.nselect for _ in columns]
        m = 0
        for i in range(snap.natoms):
            if not snap.aselect[i]:
                continue
            for j, col in enumerate(columns):
                values[j][m] = snap.atoms[i][col]
            m += 1
        return values[0] if len(lst) == 1 else values

    def newcolumn(self, name: str):
        ncol = self.snaps[0].atoms.shape[1]
        self.map(ncol + 1, name)
        for snap in self.snaps:
            atoms = snap.atoms
            newatoms = np.zeros((snap.natoms, ncol + 1), dtype=float)
            newatoms[:, 0:ncol] = atoms
            snap.atoms = newatoms

    def compare_time(self, a: Snap, b: Snap):
        return -1 if a.time < b.time else (1 if a.time > b.time else 0)

    def cull(self):
        i = 1
        while i < len(self.snaps):
            if self.snaps[i].time == self.snaps[i - 1].time:
                del self.snaps[i]
            else:
                i += 1

    def iterator(self, flag):
        start = 0
        if flag:
            start = self.iterate + 1
        for i in range(start, self.nsnaps):
            if self.snaps[i].tselect:
                self.iterate = i
                return i, self.snaps[i].time, 1
        return 0, 0, -1

    def viz(self, index, flag=0):
        if not flag:
            isnap = index
        else:
            times = self.time()
            n = len(times)
            i = 0
            while i < n:
                if times[i] > index:
                    break
                i += 1
            isnap = i - 1

        snap = self.snaps[isnap]

        time = snap.time
        box = [snap.xlo, snap.ylo, snap.zlo, snap.xhi, snap.yhi, snap.zhi]
        idc = self.names["id"]
        typ = self.names[self.atype]
        x = self.names["x"]; y = self.names["y"]; z = self.names["z"]

        atoms = []
        for i in range(snap.natoms):
            if not snap.aselect[i]:
                continue
            atom = snap.atoms[i]
            atoms.append([atom[idc], atom[typ], atom[x], atom[y], atom[z]])

        # bonds/tris/lines omitted as not used in this pipeline (retained API)
        bonds = []
        tris = []
        lines = []
        if self.triflag == 1:
            tris = self.trilist
        if self.lineflag == 1:
            lines = self.linelist

        return time, box, atoms, bonds, tris, lines

    def findtime(self, n: int) -> int:
        for i in range(self.nsnaps):
            if self.snaps[i].time == n:
                return i
        raise Exception(f"no step {n} exists")

    def maxbox(self):
        xlo = ylo = zlo = None
        xhi = yhi = zhi = None
        for snap in self.snaps:
            if not snap.tselect:
                continue
            xlo = snap.xlo if xlo is None or snap.xlo < xlo else xlo
            xhi = snap.xhi if xhi is None or snap.xhi > xhi else xhi
            ylo = snap.ylo if ylo is None or snap.ylo < ylo else ylo
            yhi = snap.yhi if yhi is None or snap.yhi > yhi else yhi
            zlo = snap.zlo if zlo is None or snap.zlo < zlo else zlo
            zhi = snap.zhi if zhi is None or snap.zhi > zhi else zhi
        return [xlo, ylo, zlo, xhi, yhi, zhi]

    def maxtype(self) -> int:
        icol = self.names["type"]
        maxv = 0
        for snap in self.snaps:
            if not snap.tselect:
                continue
            atoms = snap.atoms
            for i in range(snap.natoms):
                if not snap.aselect[i]:
                    continue
                if atoms[i][icol] > maxv:
                    maxv = atoms[i][icol]
        return int(maxv)


class tselect:
    def __init__(self, data: dump):
        self.data = data

    def all(self, **kwargs):
        outputfl = kwargs.get("output", True)
        data = self.data
        for snap in data.snaps:
            snap.tselect = 1
        data.nselect = len(data.snaps)
        data.aselect.all()
        _println(f"{data.nselect} snapshots selected out of {data.nsnaps}", quiet=not outputfl)

    def one(self, n: int):
        data = self.data
        for snap in data.snaps:
            snap.tselect = 0
        i = data.findtime(n)
        data.snaps[i].tselect = 1
        data.nselect = 1
        data.aselect.all()
        _println(f"{data.nselect} snapshots selected out of {data.nsnaps}")

    def none(self):
        data = self.data
        for snap in data.snaps:
            snap.tselect = 0
        data.nselect = 0
        _println(f"{data.nselect} snapshots selected out of {data.nsnaps}")

    def skip(self, n: int):
        data = self.data
        count = n - 1
        for snap in data.snaps:
            if not snap.tselect:
                continue
            count += 1
            if count == n:
                count = 0
                continue
            snap.tselect = 0
            data.nselect -= 1
        data.aselect.all()
        _println(f"{data.nselect} snapshots selected out of {data.nsnaps}")

    def test(self, teststr: str):
        data = self.data
        snaps = data.snaps
        cmd = "flag = " + teststr.replace("$t", "snaps[i].time")
        ccmd = compile(cmd, "", "exec")
        for i in range(data.nsnaps):
            if not snaps[i].tselect:
                continue
            local_env = {"snaps": snaps, "i": i}
            exec(ccmd, {}, local_env)
            flag = local_env.get("flag", False)
            if not flag:
                snaps[i].tselect = 0
                data.nselect -= 1
        data.aselect.all()
        _println(f"{data.nselect} snapshots selected out of {data.nsnaps}")


class aselect:
    def __init__(self, data: dump):
        self.data = data

    def all(self, *args):
        data = self.data
        if len(args) == 0:
            for snap in data.snaps:
                if not snap.tselect:
                    continue
                snap.aselect[:] = 1
                snap.nselect = snap.natoms
        else:
            n = data.findtime(args[0])
            snap = data.snaps[n]
            snap.aselect[:] = 1
            snap.nselect = snap.natoms

    def test(self, teststr: str, *args):
        data = self.data
        pattern = r"\$\w*"
        lst = re.findall(pattern, teststr)
        for item in lst:
            name = item[1:]
            column = data.names[name]
            insert = f"snap.atoms[i][{column}]"
            teststr = teststr.replace(item, insert)
        cmd = "flag = " + teststr
        ccmd = compile(cmd, "", "exec")

        if len(args) == 0:
            for snap in data.snaps:
                if not snap.tselect:
                    continue
                for i in range(snap.natoms):
                    if not snap.aselect[i]:
                        continue
                    local_env = {"snap": snap, "i": i}
                    exec(ccmd, {}, local_env)
                    if not local_env.get("flag", False):
                        snap.aselect[i] = 0
                        snap.nselect -= 1
            for i in range(data.nsnaps):
                if data.snaps[i].tselect:
                    _println(f"{data.snaps[i].nselect} atoms of {data.snaps[i].natoms} selected in first step {data.snaps[i].time}")
                    break
            for i in range(data.nsnaps - 1, -1, -1):
                if data.snaps[i].tselect:
                    _println(f"{data.snaps[i].nselect} atoms of {data.snaps[i].natoms} selected in last step {data.snaps[i].time}")
                    break
        else:
            n = data.findtime(args[0])
            snap = data.snaps[n]
            for i in range(snap.natoms):
                if not snap.aselect[i]:
                    continue
                local_env = {"snap": snap, "i": i}
                exec(ccmd, {}, local_env)
                if not local_env.get("flag", False):
                    snap.aselect[i] = 0
                    snap.nselect -= 1

# -----------------------------------------------------------------------------
# VTK writing backends
# -----------------------------------------------------------------------------

def generate_filename(root: str, fileNos: List[int], n: int) -> Tuple[str, str, str]:
    num = fileNos[n]
    if num < 10:
        s = f"{root}000{num}"
    elif num < 100:
        s = f"{root}00{num}"
    elif num < 1000:
        s = f"{root}0{num}"
    else:
        s = f"{root}{num}"
    return s + ".vtk", s + "_boundingBox.vtk", s + "_walls.vtk"


def _vtk_dtype_from_value(v: Any) -> str:
    t = type(v)
    if t in (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        return "int"
    return "float"


class LegacyVTKWriter:
    """
    Pure-Python VTK legacy writer (ASCII/BINARY) compatible with the original code.
    """

    def __init__(self, data: dump, fmt: str = "ascii", quiet: bool = False):
        self.data = data
        fmt = fmt.lower()
        if fmt not in ("ascii", "binary"):
            raise ValueError("--format must be 'ascii' or 'binary'")
        self.fmt = fmt
        self.quiet = quiet

    # ------------------------------------------------------------------

    def one(self, *args):
        file = "tmp.vtk" if len(args) == 0 else (args[0] if args[0].endswith(".vtk") else args[0] + ".vtk")
        n = flag = 0
        which, time, flag = self.data.iterator(flag)
        time, box, atoms, bonds, tris, lines = self.data.viz(which)
        _println(time, end=" ", quiet=self.quiet)

        if len(tris):
            self.surface(tris)

        allatoms = [atom for atom in atoms]
        while True:
            which, time, flag = self.data.iterator(flag)
            if flag == -1:
                break
            time, box, atoms, bonds, tris, lines = self.data.viz(which)
            for atom in atoms:
                allatoms.append(atom)
            _println(time, end=" ", quiet=self.quiet)
            n += 1

        self.particle(file, allatoms)
        _println(f"\nwrote {n} snapshots to {file} in VTK format", quiet=self.quiet)

    # ------------------------------------------------------------------

    def many(self, *args):
        root = "tmp" if len(args) == 0 else args[0]
        surfflag = 0
        n = flag = 0
        while True:
            which, time, flag = self.data.iterator(flag)
            if flag == -1:
                break
            time, box, atoms, bonds, tris, lines = self.data.viz(which)
            if surfflag == 0 and len(tris):
                surfflag = 1
                self.surface(tris)
            if n < 10:
                file = root + f"000{n}.vtk"
            elif n < 100:
                file = root + f"00{n}.vtk"
            elif n < 1000:
                file = root + f"0{n}.vtk"
            else:
                file = root + f"{n}.vtk"
            self.particle(file, atoms)
            _println(time, end=" ", quiet=self.quiet)
            n += 1
        _println(f"\nwrote {n} snapshots in VTK format", quiet=self.quiet)

    # ------------------------------------------------------------------

    def manyGran(self, *args, **kwargs):
        outputfl = kwargs.get("output", True)
        fileNos = kwargs.get("fileNos", list(range(len(self.data.snaps))))
        root = "tmp" if len(args) == 0 else args[0]

        surfflag = 0
        n = flag = 0
        while True:
            which, time, flag = self.data.iterator(flag)
            if flag == -1:
                break
            time, box, atoms, bonds, tris, lines = self.data.viz(which)

            xlo = self.data.snaps[n].xlo
            xhi = self.data.snaps[n].xhi
            ylo = self.data.snaps[n].ylo
            yhi = self.data.snaps[n].yhi
            zlo = self.data.snaps[n].zlo
            zhi = self.data.snaps[n].zhi

            atoms_full = self.data.snaps[n].atoms
            names = self.data.names

            if surfflag == 0 and len(tris):
                surfflag = 1
                self.surface(tris)

            file, file_bb, _file_walls = generate_filename(root, fileNos, n)

            self.boundingBox(file_bb, xlo, xhi, ylo, yhi, zlo, zhi)
            try:
                nvalues = len(self.data.snaps[0].atoms[0])
            except Exception:
                nvalues = 0

            self.particleGran(file, atoms_full, names, nvalues)
            if outputfl:
                _println(time, end=" ", quiet=self.quiet)
            n += 1

        if outputfl:
            _println(f"\nwrote {n} granular snapshots in VTK format", quiet=self.quiet)

    # ------------------------------------------------------------------
    # Writing helpers (ASCII and BINARY)
    # ------------------------------------------------------------------

    def _open(self, file: str):
        return open(file, "wb" if self.fmt == "binary" else "w", newline="")

    def _write_header(self, f, comment: str, dataset: str):
        if self.fmt == "binary":
            f.write(b"# vtk DataFile Version 2.0\n")
            f.write((comment + "\n").encode("ascii"))
            f.write(b"BINARY\n")
            f.write((f"DATASET {dataset}\n").encode("ascii"))
        else:
            print("# vtk DataFile Version 2.0", file=f)
            print(comment, file=f)
            print("ASCII", file=f)
            print(f"DATASET {dataset}", file=f)

    def _writeln(self, f, s: str):
        if self.fmt == "binary":
            f.write((s + "\n").encode("ascii"))
        else:
            print(s, file=f)

    def _write_floats(self, f, arr: Iterable[float]):
        if self.fmt == "binary":
            for v in arr:
                f.write(struct.pack(">f", float(v)))
            f.write(b"\n")
        else:
            line = []
            cnt = 0
            for v in arr:
                line.append(f"{float(v):.9g}")
                cnt += 1
                if cnt == 9:
                    self._writeln(f, " ".join(line))
                    line = []
                    cnt = 0
            if line:
                self._writeln(f, " ".join(line))

    def _write_ints(self, f, arr: Iterable[int]):
        if self.fmt == "binary":
            for v in arr:
                f.write(struct.pack(">i", int(v)))
            f.write(b"\n")
        else:
            line = []
            cnt = 0
            for v in arr:
                line.append(str(int(v)))
                cnt += 1
                if cnt == 16:
                    self._writeln(f, " ".join(line))
                    line = []
                    cnt = 0
            if line:
                self._writeln(f, " ".join(line))

    # ------------------------------------------------------------------

    def particle(self, file: str, atoms: List[List[float]]):
        with self._open(file) as f:
            self._write_header(f, "Generated by lpp (legacy)", "POLYDATA")
            npts = len(atoms)
            self._writeln(f, f"POINTS {npts} float")
            coords = []
            for atom in atoms:
                coords.extend([atom[2], atom[3], atom[4]])
            self._write_floats(f, coords)

            self._writeln(f, f"VERTICES {npts} {2*npts}")
            ints = []
            for i in range(npts):
                ints.extend([1, i])
            self._write_ints(f, ints)

            self._writeln(f, f"POINT_DATA {npts}")
            self._writeln(f, "SCALARS atom_type int 1")
            self._writeln(f, "LOOKUP_TABLE default")
            types_arr = [int(atom[1]) for atom in atoms]
            self._write_ints(f, types_arr)

    def boundingBox(self, file: str, xlo, xhi, ylo, yhi, zlo, zhi):
        with self._open(file) as f:
            self._write_header(f, "Generated by lpp (legacy)", "RECTILINEAR_GRID")
            self._writeln(f, "DIMENSIONS 2 2 2")
            self._writeln(f, "X_COORDINATES 2 float")
            self._write_floats(f, [xlo, xhi])
            self._writeln(f, "Y_COORDINATES 2 float")
            self._write_floats(f, [ylo, yhi])
            self._writeln(f, "Z_COORDINATES 2 float")
            self._write_floats(f, [zlo, zhi])

    def surface(self, tris: List[List[float]]):
        """
        Write triangles to SURF{type}.vtk files (ASCII/BINARY).
        """
        # Determine number of types
        ntypes = tris[-1][1]
        for i in range(int(ntypes)):
            itype = i + 1
            v = {}
            nvert = 0
            tri_list = []
            for tri in tris:
                if tri[1] != itype:
                    continue
                v1 = (tri[2], tri[3], tri[4])
                v2 = (tri[5], tri[6], tri[7])
                v3 = (tri[8], tri[9], tri[10])
                for vert in (v1, v2, v3):
                    if vert not in v:
                        v[vert] = nvert
                        nvert += 1
                tri_list.append((v1, v2, v3))

            vinverse = {idx: key for key, idx in v.items()}
            filename = f"SURF{itype}.vtk"
            with self._open(filename) as f:
                self._write_header(f, "Generated by lpp (legacy)", "POLYDATA")
                self._writeln(f, f"POINTS {nvert} float")
                coords = []
                for idx in range(nvert):
                    tpl = vinverse[idx]
                    coords.extend([tpl[0], tpl[1], tpl[2]])
                self._write_floats(f, coords)

                ntri = len(tri_list)
                self._writeln(f, f"POLYGONS {ntri} {4*ntri}")
                ints = []
                for tri in tri_list:
                    i1 = v[tri[0]]; i2 = v[tri[1]]; i3 = v[tri[2]]
                    ints.extend([3, i1, i2, i3])
                self._write_ints(f, ints)

                self._writeln(f, "")
                self._writeln(f, f"CELL_DATA {ntri}")
                self._writeln(f, f"POINT_DATA {nvert}")

    # ------------------------------------------------------------------

    def particleGran(self, file: str, atoms_full: Optional[np.ndarray], names: Dict[str, int], n_values: int):
        # If no atoms are present
        if atoms_full is None:
            atoms_full = np.zeros((0, 0), dtype=float)

        scalars, vectors = findScalarsAndVectors(names)

        with self._open(file) as f:
            self._write_header(f, "Generated by lpp (legacy)", "POLYDATA")
            npts = atoms_full.shape[0] if atoms_full is not None else 0
            self._writeln(f, f"POINTS {npts} float")
            coords = []
            if npts > 0:
                xidx = vectors["x"]
                coords = atoms_full[:, xidx:xidx+3].reshape(-1).tolist()
            self._write_floats(f, coords)

            self._writeln(f, f"VERTICES {npts} {2*npts}")
            ints = []
            for i in range(npts):
                ints.extend([1, i])
            self._write_ints(f, ints)

            self._writeln(f, f"POINT_DATA {npts}")
            if npts == 0:
                self._writeln(f, "")
                return

            # VECTORS (skip coordinates key 'x')
            for key, start_idx in vectors.items():
                if key == "x":
                    continue
                self._writeln(f, f"VECTORS {key} float")  # write as float
                vec_data = atoms_full[:, start_idx:start_idx+3].reshape(-1).tolist()
                self._write_floats(f, vec_data)

            # SCALARS
            for key, idx in scalars.items():
                # infer type; write ints as int, floats as float
                sample = atoms_full[0, idx] if npts > 0 else 0.0
                scalar_type = "int" if isinstance(sample, (np.integer, int)) else "float"
                self._writeln(f, f"SCALARS {key} {scalar_type} 1")
                self._writeln(f, "LOOKUP_TABLE default")
                if scalar_type == "int":
                    vals = [int(v) for v in atoms_full[:, idx].tolist()]
                    self._write_ints(f, vals)
                else:
                    vals = atoms_full[:, idx].tolist()
                    self._write_floats(f, vals)

            self._writeln(f, "")


# ------------------------- vtk library backend -------------------------------

class VTKLibWriter:
    """
    Writer backend using the third‑party 'vtk' package (if available).
    This is optional; if import fails, caller should fall back to LegacyVTKWriter.
    """
    def __init__(self, data: dump, fmt: str = "ascii", quiet: bool = False):
        global _VTK_AVAILABLE, _vtk
        if not _VTK_AVAILABLE or _vtk is None:
            # Attempt import
            try:
                import vtk as _vtk_mod  # type: ignore
                _VTK_AVAILABLE = True
                _vtk = _vtk_mod
            except Exception:
                _VTK_AVAILABLE = False
                _vtk = None
        if not _VTK_AVAILABLE:
            raise RuntimeError("vtk backend not available")
        self.vtk = _vtk
        self.data = data
        fmt = fmt.lower()
        if fmt not in ("ascii", "binary"):
            raise ValueError("--format must be 'ascii' or 'binary'")
        self.fmt = fmt
        self.quiet = quiet

    # ------------------------------------------------------------------

    def _set_writer_mode(self, writer):
        if self.fmt == "binary":
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()

    # ------------------------------------------------------------------

    def manyGran(self, *args, **kwargs):
        outputfl = kwargs.get("output", True)
        fileNos = kwargs.get("fileNos", list(range(len(self.data.snaps))))
        root = "tmp" if len(args) == 0 else args[0]

        n = flag = 0
        while True:
            which, time, flag = self.data.iterator(flag)
            if flag == -1:
                break
            time, box, atoms, bonds, tris, lines = self.data.viz(which)

            xlo = self.data.snaps[n].xlo
            xhi = self.data.snaps[n].xhi
            ylo = self.data.snaps[n].ylo
            yhi = self.data.snaps[n].yhi
            zlo = self.data.snaps[n].zlo
            zhi = self.data.snaps[n].zhi

            atoms_full = self.data.snaps[n].atoms
            names = self.data.names

            file, file_bb, _file_walls = generate_filename(root, fileNos, n)

            # Particles
            self._write_polydata_gran(file, atoms_full, names)

            # Bounding box
            self._write_rectilinear_grid(file_bb, xlo, xhi, ylo, yhi, zlo, zhi)

            if outputfl:
                _println(time, end=" ", quiet=self.quiet)
            n += 1
        if outputfl:
            _println(f"\nwrote {n} granular snapshots in VTK format", quiet=self.quiet)

    def _write_polydata_gran(self, file: str, atoms_full: Optional[np.ndarray], names: Dict[str, int]):
        vtk = self.vtk
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        pdata = poly.GetPointData()

        npts = 0 if atoms_full is None else atoms_full.shape[0]
        if npts > 0 and "x" in names:
            xidx = names["x"]
            coords = atoms_full[:, xidx:xidx+3]
            for i in range(npts):
                points.InsertNextPoint(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
        poly.SetPoints(points)
        poly.SetVerts(verts)

        scalars, vectors = findScalarsAndVectors(names)
        # VECTORS (skip coordinates)
        for key, start_idx in vectors.items():
            if key == "x":
                continue
            arr = vtk.vtkFloatArray()
            arr.SetName(key)
            arr.SetNumberOfComponents(3)
            for i in range(npts):
                v = atoms_full[i, start_idx:start_idx+3]
                arr.InsertNextTuple3(float(v[0]), float(v[1]), float(v[2]))
            pdata.AddArray(arr)

        # SCALARS
        for key, idx in scalars.items():
            # Decide int/float: use float array generally (vtk often stores as float)
            # Keep compatibility: if values are all integers, store as int.
            col = atoms_full[:, idx] if npts > 0 else np.array([], dtype=float)
            if npts > 0 and np.all(np.equal(np.mod(col, 1), 0)):
                arr = vtk.vtkIntArray()
                for v in col:
                    arr.InsertNextValue(int(v))
            else:
                arr = vtk.vtkFloatArray()
                for v in col:
                    arr.InsertNextValue(float(v))
            arr.SetName(key)
            pdata.AddArray(arr)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(file)
        writer.SetInputData(poly)
        self._set_writer_mode(writer)
        writer.Write()

    def _write_rectilinear_grid(self, file: str, xlo, xhi, ylo, yhi, zlo, zhi):
        vtk = self.vtk
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(2, 2, 2)

        def _vtk_float_array(vals):
            arr = vtk.vtkFloatArray()
            for v in vals:
                arr.InsertNextValue(float(v))
            return arr

        grid.SetXCoordinates(_vtk_float_array([xlo, xhi]))
        grid.SetYCoordinates(_vtk_float_array([ylo, yhi]))
        grid.SetZCoordinates(_vtk_float_array([zlo, zhi]))

        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileName(file)
        writer.SetInputData(grid)
        self._set_writer_mode(writer)
        writer.Write()


# -----------------------------------------------------------------------------
# Shared helpers from original vtk.py
# -----------------------------------------------------------------------------

def typestr(o):
    return type(o).__name__

def findScalarsAndVectors(names: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    vectors = {}
    scalars = {}

    # reverse dictionary {position: name}
    indices: Dict[int, str] = {}
    for name, idx in names.items():
        indices[idx] = name

    # fill missing indices
    if indices:
        for i in range(max(indices) + 1):
            if i not in indices:
                indices[i] = ""

    regvx = re.compile(".*x$")
    regvy = re.compile(".*y$")
    regvz = re.compile(".*z$")
    regf = re.compile(r"f_.*\[\d+\]")
    regc = re.compile(r"c_.*\[\d+\]")
    regv = re.compile(r"v_.*\[\d+\]")

    i = 0
    max_i = max(indices) if indices else -1
    while i <= max_i:
        namei = indices[i]

        # xyz triplet?
        if i + 2 <= max_i and regvx.match(indices[i]) and regvy.match(indices[i + 1]) and regvz.match(indices[i + 2]):
            newname = indices[i] if len(indices[i]) == 1 else indices[i][:-1]
            vectors[newname] = i
            i += 3
            continue

        # f_/c_/v_ triplet with bracket indices
        if regf.match(indices[i]) or regc.match(indices[i]) or regv.match(indices[i]):
            name = indices[i]
            try:
                number = int(name.split("[")[1].split("]")[0])
            except Exception:
                number = 0
            base = name.split("[")[0]
            nextName = f"{base}[{number + 1}]"
            nextButOneName = f"{base}[{number + 2}]"
            newname = name[2:-(len(name.split("[")[1]) + 1)]  # strip prefix and [n]

            if i + 2 <= max_i and indices[i + 1] == nextName and indices[i + 2] == nextButOneName:
                vectors[newname] = i
                i += 3
                continue
            else:
                scalars[newname] = i
                i += 1
                continue

        if namei != "":
            scalars[namei] = i
        i += 1

    if "x" not in vectors:
        _println("vector x y z has to be contained in dump file. please change liggghts input script accordingly.", file=sys.stderr)
        sys.exit(1)

    return scalars, vectors


# -----------------------------------------------------------------------------
# Top-level LPP orchestrator (port of lpp.py) with Python3 + new options
# -----------------------------------------------------------------------------

class LPP:
    def __init__(self, filelist: List[str], **kwargs):
        # Defaults
        self.cpunum = max(1, mp.cpu_count())
        self.chunksize = 8
        self.overwrite = True
        self.debugMode = False
        self.output = True
        self.format = kwargs.get("format", "ascii").lower()  # ascii|binary
        self.backend = kwargs.get("backend", "legacy").lower()  # legacy|vtk

        # Parse options
        if "chunksize" in kwargs:
            try:
                cs = int(kwargs["chunksize"])
                if cs <= 0:
                    raise ValueError
                self.chunksize = cs
            except Exception:
                raise ValueError("Invalid or no argument given for --chunksize")

        if "cpunum" in kwargs:
            try:
                cpu = int(kwargs["cpunum"])
                if cpu <= 0 or cpu > self.cpunum:
                    raise ValueError
                self.cpunum = cpu
            except Exception:
                raise ValueError("Invalid or no argument given for --cpunum")

        if kwargs.get("no_overwrite", False):
            self.overwrite = False

        if kwargs.get("debug", False):
            self.debugMode = True

        if kwargs.get("quiet", False):
            self.output = False
            self.debugMode = False

        outroot = kwargs.get("output_root", "")

        if self.output:
            _println("starting LIGGGHTS memory-optimized parallel post processing")
            _println(f"chunksize: {self.chunksize}  -> {self.chunksize} files are processed per chunk. If you run out of memory reduce chunksize.")
            _println(f"backend: {self.backend}, format: {self.format}")

        if self.debugMode:
            _println(f"master pid: {os.getpid()}")

        # Build file list (glob expansion on Windows if needed)
        flist: List[str] = []
        if os.name == "nt":
            for item in filelist:
                if "*" in item:
                    flist.extend(glob.glob(item))
                else:
                    flist.append(item)
        else:
            flist = list(filelist)

        listlen = len(flist)
        if listlen == 0:
            raise Exception("no dump file specified")
        if listlen == 1 and self.overwrite is False:
            raise Exception("Cannot process single dump files with --no-overwrite.")

        if self.output:
            _println(f"Working with {self.cpunum} processes...")

        # Slice into chunks
        slices: List[List[str]] = []
        residual_present = int(bool(listlen - floor(listlen / self.chunksize) * self.chunksize))
        for i in range(int(floor(listlen / self.chunksize)) + residual_present):
            slices.append(flist[i * self.chunksize : (i + 1) * self.chunksize])

        dumpInput = [{
            "filelist": slices[i],
            "debugMode": self.debugMode,
            "output": outroot,
            "overwrite": self.overwrite,
            "format": self.format,
            "backend": self.backend,
        } for i in range(len(slices))]

        numberOfRuns = len(dumpInput)
        i = 0
        while i < len(dumpInput):
            if self.output:
                _println(f"calculating chunks {i+1}-{min(i+self.cpunum, numberOfRuns)} of {numberOfRuns}")
            if self.debugMode:
                _println(f"input of this map: {dumpInput[i:i+self.cpunum]}")
            with mp.Pool(processes=self.cpunum) as pool:
                # Windows fix: do not pass an oversized timeout; block indefinitely.
                pool.map_async(lpp_worker, dumpInput[i:i+self.cpunum]).get()
            i += self.cpunum

        _println(f"wrote {listlen} granular snapshots in VTK format")

def lpp_worker(input_dict: Dict[str, Any]) -> int:
    flist: List[str] = input_dict["filelist"]
    debugMode: bool = input_dict["debugMode"]
    outfileName: str = input_dict["output"]
    overwrite: bool = input_dict["overwrite"]
    fmt: str = input_dict.get("format", "ascii").lower()
    backend: str = input_dict.get("backend", "legacy").lower()

    # generate output root
    # derive sensible output root from first input file
    inpath = flist[0]
    base = os.path.basename(inpath)
    # strip .gz then one more extension if present
    base = re.sub(r'\.gz$', '', base)
    base = re.sub(r'\.[^.]+$', '', base)
    # if ends with -12345 or _12345, keep the separator and drop the digits
    m = re.match(r'^(.*?)([-_])\d+$', base)
    default_root = (m.group(1) + m.group(2)) if m else (base + "-")

    if outfileName == "":
        granName = default_root
    elif outfileName.endswith(("/", "\\")):  # support / and \ as "folder/"
        granName = outfileName + default_root
    else:
        granName = outfileName

    shortFlist: List[str] = []
    if overwrite:
        shortFlist = flist
    else:
        # check by time stamp in second line of dump file
        for fpath in flist:
            try:
                if fpath.endswith(".gz"):
                    ff = gzip.open(fpath, "rt")
                else:
                    ff = open(fpath, "r")
                ff.readline()             # ITEM: TIMESTEP
                tline = ff.readline()
                time_val = int(tline.strip().split()[0])
                ff.close()
            except Exception:
                continue
            filename, _, _ = generate_filename(granName, [time_val], 0)
            if not os.path.isfile(filename):
                shortFlist.append(fpath)

    try:
        d = dump({"filelist": shortFlist, "debugMode": debugMode})
        # choose writer backend
        if backend == "vtk":
            try:
                writer = VTKLibWriter(d, fmt=fmt, quiet=not debugMode)
            except Exception:
                writer = LegacyVTKWriter(d, fmt=fmt, quiet=not debugMode)
        else:
            writer = LegacyVTKWriter(d, fmt=fmt, quiet=not debugMode)

        if debugMode:
            _println(f"\nfileNums: {d.fileNums}\n")
        writer.manyGran(granName, fileNos=d.fileNums, output=debugMode)
    except KeyboardInterrupt:
        raise
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="LIGGGHTS memory-optimized parallel post processing -> VTK (Python 3 single file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dump_files", nargs="+", help="dump file(s) or glob patterns")
    p.add_argument("-o", "--output", dest="output_root", default="", help="output file name/prefix (default: liggghts + timestep)")
    p.add_argument("--chunksize", type=int, default=8, help="files per chunk")
    p.add_argument("--cpunum", type=int, default=max(1, mp.cpu_count()), help="number of worker processes")
    p.add_argument("--no-overwrite", dest="no_overwrite", action="store_true", help="do not overwrite pre-generated files")
    p.add_argument("--debug", action="store_true", help="enable debug output")
    p.add_argument("--quiet", action="store_true", help="suppress output (overrides --debug)")
    p.add_argument("--format", choices=["ascii", "binary"], default="ascii", help="VTK legacy output format")
    p.add_argument("--backend", choices=["legacy", "vtk"], default="legacy", help="writer backend ('vtk' uses python-vtk if available)")
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        LPP(args.dump_files,
            chunksize=args.chunksize,
            cpunum=args.cpunum,
            no_overwrite=args.no_overwrite,
            debug=args.debug,
            quiet=args.quiet,
            output_root=args.output_root,
            format=args.format,
            backend=args.backend,
        )
    except KeyboardInterrupt:
        _println("aborted by user", file=sys.stderr)
    except BaseException as e:
        _println(f"aborting due to errors: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Windows support for multiprocessing + avoid large-timeout bug
    mp.freeze_support()
    main()

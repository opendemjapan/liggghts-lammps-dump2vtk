# lpp.py — Unified LIGGGHTS Post Processing (Python 3)

> **LIGGGHTS / LAMMPS のダンプ (`dump.*` / `dump.*.gz`) を VTK Legacy `.vtk` へ高速一括変換するツールである.**  
> 旧 Pizza.py (`lpp.py` / `vtk.py` / `dump.py`) を Python 3 に統合移植し, ASCII / BINARY 切替と `vtk` バックエンドを備えた単一ファイル版である.  
> **lpp / pizza** を基にした派生物である.

---

## 目次
- [特徴](#特徴)
- [ダウンロード](#ダウンロード)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [主なオプション](#主なオプション)
- [使用例](#使用例)
- [出力について](#出力について)
- [バックエンドの切替](#バックエンドの切替)
- [性能チューニング](#性能チューニング)
- [環境変数](#環境変数)
- [互換性](#互換性)
- [既知の事項](#既知の事項)
- [謝辞](#謝辞)
- [ライセンス](#ライセンス)

---

## 特徴

- **単一ファイル**: 旧 `lpp.py` / `vtk.py` / `dump.py` を 1 つの Python 3 スクリプトに統合.  
- **VTK Legacy 出力**: **ASCII / BINARY (ビッグエンディアン)** を選択可能.  
- **バックエンド**: 依存無しの **legacy writer (既定)** と, `pip install vtk` で利用可能な **`vtk` バックエンド**.  
- **自動フォールバック**: `--backend vtk` 指定時に `vtk` が無い場合, 自動で legacy に切替.  
- **Pizza.py 互換の選択 / 整形**: スナップショット選択, マッピング, スケール / アンスケール, 並べ替えに対応.  
- **並列処理**: チャンク分割 + 複数プロセスで大量ステップを高速変換.  
- **上書き抑止**: 既存 `.vtk` を再生成しない `--no-overwrite` を提供.  
- **ボックスのグリッド出力**: 粒子データと併せて **RECTILINEAR_GRID** を生成.  
- **デバッグ / 静穏モード**: `--debug` / `--quiet` を備える.  

---

## ダウンロード

- **スクリプト本体 (raw)**: [`lpp.py`](./lpp.py?raw=1)  
- **最新リリース**: [`Releases`](../../releases/latest)  
- **リポジトリ一括 (ZIP)**: [`Download ZIP`](../../archive/refs/heads/main.zip)  

> GitHub 上での利用を想定した例である. リポジトリ構成に応じてパスを調整すること.  

---

## インストール

**必須**
```bash
python -m pip install numpy
```

**オプション (`--backend vtk` を用いる場合)**
```bash
python -m pip install vtk
```

> 環境により `pip` の代わりに `pip3` を用いること.  

---

## クイックスタート

```bash
# ASCII (既定) で VTK を出力
python lpp.py dump.*.gz -o out/liggghts --format ascii

# バイナリ VTK (ビッグエンディアン)
python lpp.py dump.* -o out/liggghts --format binary

# vtk バックエンドを用いる (無ければ自動で legacy にフォールバック)
python lpp.py dump.* -o out/liggghts --backend vtk --format binary
```

---

## 主なオプション

| オプション | 説明 | 既定値 |
|---|---|---|
| `-o, --output ROOT` | 出力ファイル名 / プレフィックス. タイムステップは自動付与. | `liggghts` + timestep |
| `--chunksize N` | 1 プロセスがまとめて処理するファイル数. | `8` |
| `--cpunum N` | ワーカープロセス数. CPU コア数を自動推定. | マシンのコア数 |
| `--no-overwrite` | 既存 `.vtk` がある場合は生成しない. | 無効 |
| `--debug` | 進行や詳細ログを表示. | 無効 |
| `--quiet` | 最小限の出力. `--debug` を上書き. | 無効 |
| `--format {ascii,binary}` | VTK Legacy 形式 (ASCII / BINARY). | `ascii` |
| `--backend {legacy,vtk}` | 出力バックエンド (依存無し / `vtk` ライブラリ). | `legacy` |
| `--help` | ヘルプを表示. | - |

> **注**: BINARY は VTK Legacy 仕様に従い **ビッグエンディアン** で出力する.  

---

## 使用例

### 1) `.gz` を含む大量ダンプを一括変換 (既存をスキップ)
```bash
python lpp.py "dump.*.gz" -o vtk/out --format binary --cpunum 8 --chunksize 16 --no-overwrite
```

### 2) `vtk` バックエンドでポリデータ / ボックスを出力
```bash
python lpp.py dump.liggghts.* -o vtk/out --backend vtk --format ascii
```

### 3) 静穏モードで最小出力
```bash
python lpp.py dump* -o vtk/out --quiet
```

---

## 出力について

- 粒子は **VTK Legacy `.vtk` (POLYDATA など)** としてタイムステップ毎に出力される.  
  例) `out/liggghts.100000.vtk`, `out/liggghts.110000.vtk`, …  
- **境界ボックス (RECTILINEAR_GRID)** も粒子ファイルと同名で生成される.  
- **サーフェス三角形** の書き出しに対応 (ASCII / バイナリ両対応).  
- タイムステップは読込後に **昇順ソート** し, **重複は自動で間引く**.  

---

## バックエンドの切替

- **`legacy` (既定)**: 外部依存無し. VTK Legacy を自前で書き出す (ASCII / BINARY).  
- **`vtk`**: `pip install vtk` が必要. `vtkPolyDataWriter` / `vtkRectilinearGridWriter` を用い, `--format` の指定に従う.  
- `--backend vtk` 指定時にモジュールが見つからない場合, **自動的に legacy へフォールバック** する.  

---

## 性能チューニング

- **並列度**: `--cpunum` でワーカー数を指定. 未指定時は CPU コア数を用いる.  
- **I / O バランス**: `--chunksize` を増やすと I / O 集約で効率が上がる一方, メモリ使用量が増える.  
- **再実行抑止**: `--no-overwrite` で既存 `.vtk` をスキップし, 再実行を短縮する.  

---

## 環境変数

- `PIZZA_GUNZIP`: `.gz` 解凍コマンドを指定する (既定: `gunzip`).  
  例) `PIZZA_GUNZIP="pigz -d"` など.  

---

## 互換性

- **Python**: 3.8 以上.  
- **依存**: NumPy が必須. `--backend vtk` 使用時は `vtk` が必要.  

---

## 既知の事項

- VTK Legacy BINARY は **ビッグエンディアン固定** である. ビューア設定に注意すること.  
- 旧 Pizza.py 由来のカラム名 (`x / xs / xu`, `y / ys / yu`, `z / zs / zu` など) を **可能な範囲で自動解釈** する.  
- GZIP ダンプ (`*.gz`) は Python 側で透過読込する.  

---

## 謝辞

 本ツールは **Pizza.py / lpp** 系ツール群 (`dump.py`, `vtk.py`, `lpp.py`) を基に再構成したものである.  
 LIGGGHTS / LAMMPS コミュニティと Pizza.py の貢献に感謝する.  
 ( **lpp / pizza を基に作成** )

---

## ライセンス

 本プロジェクトは **GNU General Public License** (Pizza.py の元ライセンスに整合) で配布する.  
 派生元が GPL であるため, 本ツールも **GPL-2.0-or-later** とする.  

```
SPDX-License-Identifier: GPL-2.0-or-later

Copyright (C) 20XX-20XX <Your Name>
This program is a derivative work of Pizza.py tools (dump.py / vtk.py / lpp.py).
Copyright (C) 2003-2010 Sandia Corporation and others.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
```

> 著作権者名と年は適宜置換すること. Pizza.py 由来の著作権表示と告知文は `LICENSE` と該当ソースのヘッダに **保持** すること.  

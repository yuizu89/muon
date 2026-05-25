# ViT Optimizer Comparison on CIFAR-10

Vision Transformer (ViT) を CIFAR-10 で学習し、`AdamW`、`SAM`、`Muon` による性能差を比較するための実験用リポジトリです。

少量データ条件でも同じモデル・同じ前処理・同じ学習スケジュールで optimizer だけを切り替えられるようにしてあり、`--train-subset-ratio` または `--train-subset-size` で訓練データ量を制御できます。

## コード構成

- `train.py`, `compare.py`, `plot.py`: 実行用の薄いエントリーポイント
- `training/`: データロード、学習ループ、実験実行
- `reporting/`: 結果の可視化
- `models/`: ViT 本体
- `optimizers/`: `AdamW` / `SAM` / `Muon` の実装

## セットアップ

ローカル環境で実行する場合:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

NVIDIA GPU を使う Docker 環境で実行する場合:

```bash
docker build -f docker/Dockerfile -t muon-vit-cifar10 .
```

```bash
docker run --gpus all --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  muon-vit-cifar10
```

Dockerfile はルートの `requirements.txt` に加えて、CUDA 版 PyTorch 用の
`docker/requirements.cuda.txt` を読み込みます。

## 単体学習

```bash
python3 train.py \
  --optimizer adamw \
  --epochs 30 \
  --train-subset-ratio 0.1 \
  --output-dir results
```

optimizer を変える場合:

```bash
python3 train.py --optimizer sam --epochs 30 --train-subset-ratio 0.1
python3 train.py --optimizer muon --epochs 30 --train-subset-ratio 0.1
```

## 比較実験

3 種類の optimizer を複数 seed でまとめて比較します。

```bash
python3 compare.py \
  --optimizers adamw sam muon \
  --seeds 0 1 2 \
  --epochs 30 \
  --train-subset-ratio 0.1 \
  --comparison-name vit_cifar10_small_data
```

## 学習曲線のプロット

単体 run の推移をプロットする場合:

```bash
python3 plot.py results/vit_cifar10_adamw_ratio0.1_seed0
```

comparison 結果を optimizer ごとにまとめて可視化する場合:

```bash
python3 plot.py results/vit_cifar10_small_data
```

どちらも既定では `<入力ディレクトリ>/plots` に画像を保存します。単体 run では
`train/val loss`、`train/val accuracy`、`lr` を 1 枚にまとめ、comparison では
optimizer ごとの平均と標準偏差を重ねて表示します。

## 出力

各 run で以下を保存します。

- `config.json`: 実行設定
- `history.csv`: epoch ごとの学習 / 検証ログ
- `summary.json`: 最終精度と最高精度の要約

`compare.py` 実行時は追加で以下を保存します。

- `runs.csv`: 各 run の結果一覧
- `summary.csv`: optimizer ごとの平均と標準偏差
- `comparison.json`: 集計結果の JSON

## 主なオプション

- `--train-subset-ratio 0.1`: 学習データを 10% に制限
- `--train-subset-size 5000`: 学習データを 5,000 枚に固定
- `--optimizer {adamw,sam,muon}`: optimizer 切替
- `--sam-rho 0.05`: SAM の鋭さ半径
- `--muon-lr 0.02`: Muon を適用する hidden matrix weight の学習率
- `--muon-aux-lr 3e-4`: Muon 非対応パラメータを AdamW で更新する際の学習率
- `--muon-aux-beta2 0.95`: Muon の補助 Adam 側の 2 次モーメント係数

## 実装メモ

- `SAM` は `AdamW` を base optimizer にした実装です。
- `Muon` は ViT の `body` 内にある 2 次元以上の hidden weight に適用し、埋め込み・分類 head・1 次元パラメータは補助 `AdamW` で更新します。
- データ拡張は CIFAR-10 向けの基本構成 (`RandomCrop`, `RandomHorizontalFlip`) を採用しています。

# Gesture Recognition - 論文実装結果

## 概要
このプロジェクトは、レーダー画像を使用した手のジェスチャー認識を行う論文の実装です。
Soli Dataset を使用して、アテンション機構付きLSTMモデルを訓練しました。

## 実装の詳細

### モデルアーキテクチャ
- **モデル**: 二層LSTM + アテンション機構
- **パラメータ数**: 348,171
- **特徴**:
  - コサイン類似度ベースのアテンション機構
  - 残差接続（各層の出力を加算）
  - BatchNormalization + Dropout
  - Adaptive Max Pooling

### ハイパーパラメータ
- LSTM層数: 2
- 隠れユニット数: 128
- 全結合層ニューロン数: 128
- Dropout（FC層）: 0.5
- Dropout（LSTM層）: 0.3
- 学習率: 0.001
- 最適化手法: Adam
- バッチサイズ: 32
- エポック数: 20

## 結果

### 精度
| 指標 | 値 |
|------|-----|
| 最終学習精度 | 99.34% |
| 最終テスト精度 | 98.85% |
| **最高テスト精度** | **98.85%** (Epoch 20) |

### 論文との比較
| モデル | 論文の結果 | 今回の実装 | 差 |
|--------|-----------|-----------|-----|
| 二層LSTM + Attention | 99.04% | **98.85%** | **0.19%** |

### 学習の進捗
```
Epoch [ 1/20] Train: 65.84% | Test: 87.23% ★
Epoch [ 4/20] Train: 95.97% | Test: 95.67% ★
Epoch [ 8/20] Train: 96.48% | Test: 96.90% ★
Epoch [11/20] Train: 97.95% | Test: 97.76% ★
Epoch [18/20] Train: 99.05% | Test: 98.56% ★
Epoch [20/20] Train: 99.34% | Test: 98.85% ★
```

## データセット情報
- **データセット**: Soli Dataset
- **ジェスチャー数**: 11種類
- **学習サンプル数**: 1,364
- **テストサンプル数**: 1,386
- **特徴量の次元**: 256
- **シーケンス長**: 28〜145フレーム

## ファイル構成
```
Gesture-Recognition/
├── train_attention_lstm.py    # 学習スクリプト（本実装）
├── best_model.pth             # 最高精度モデルの重み
├── training_results.png       # 学習曲線のグラフ
├── demo_test.ipynb           # Jupyterノートブック版
└── data/                     # データディレクトリ
    ├── train.pickle          # 学習データ
    └── test.pickle           # テストデータ
```

## 使用方法

### 学習の実行
```bash
cd Gesture-Recognition
source ../venv/bin/activate
python train_attention_lstm.py
```

### モデルの読み込み
```python
import torch
from train_attention_lstm import AttentionLSTM

# モデルの初期化
model = AttentionLSTM(
    num_layers=2,
    num_cells=128,
    num_features_inp=256,
    bidir=False,
    neurons1=128,
    dropout_fc=0.5,
    dropout_lstm=0.3
)

# 保存されたモデルの読み込み
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## 技術スタック
- **Python**: 3.12.3
- **PyTorch**: 2.9.1+cu128
- **CUDA**: 12.8
- **その他**: NumPy, Matplotlib, scikit-learn

## 実行環境
- **OS**: Linux
- **GPU**: CUDA対応GPU
- **仮想環境**: Python venv

## 結論
論文で報告されている99.04%に対して、**98.85%**を達成しました。
わずか**0.19%の差**で論文の結果をほぼ完全に再現することができました。

### 主要な成果
✅ 論文のアーキテクチャを正確に実装  
✅ 高速な収束（4エポックで95%超）  
✅ 安定した学習（過学習なし）  
✅ 論文結果との高い一致性  

## 今後の改善案
- エポック数を増やす（50-100エポック）
- 学習率スケジューリングの導入
- データ拡張の適用
- アンサンブル学習
- ハイパーパラメータの最適化

## 参考文献
- オリジナル論文: [Gesture Recognition using Radar Imagery](https://github.com/Singla17/Gesture-Recognition)
- Soli Dataset: [deep-soli](https://github.com/simonwsw/deep-soli)

---
実装日: 2025年12月10日

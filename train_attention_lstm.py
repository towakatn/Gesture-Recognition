"""
Gesture Recognition using Attention-based LSTM
論文の実装：アテンション機構付きLSTMモデル
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# ランダムシード設定
def set_seed(seed=42):
    """再現性のためにランダムシードを設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

# デバイス設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
print("="*60)

# データ処理関数
def get_data(data_dict, remove_gestures=None):
    """オリジナルのLSTMノートブックから取得したデータ処理関数"""
    X = []
    y = []
    lengths = []
    
    if remove_gestures is not None:
        keys = np.zeros(11-len(remove_gestures), dtype=int)
        j = 0
        for i in range(11):
            if i not in remove_gestures:
                keys[j] = i
                j = j + 1
    else:
        keys = np.arange(11)
        
    keys = list(keys)

    for key in keys:
        gesture_data = data_dict[key]
        for i in range(len(gesture_data)):
            datapt_arr = gesture_data[i]
            datapt = None
            first = True
            for i in range(4):
                if first:
                    datapt = np.sum(datapt_arr[i], axis=2)
                    first = False
                else:
                    datapt = np.concatenate((datapt, np.sum(datapt_arr[i], axis=2)), axis=1)
            for i in range(4):
                datapt = np.concatenate((datapt, np.sum(datapt_arr[i], axis=1)), axis=1)

            X.append(datapt)
            y_temp = np.zeros(len(keys))
            y_temp[keys.index(key)] = 1
            y.append(y_temp)
            lengths.append(np.shape(datapt)[0])

    lengths = np.array(lengths)
    return X, y, lengths

# データセットクラス
class GestureDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]), self.lengths[idx]

def collate_fn(batch):
    """カスタムコレート関数でパディングを処理"""
    X_batch = [item[0] for item in batch]
    y_batch = torch.stack([item[1] for item in batch])
    lengths_batch = torch.LongTensor([item[2] for item in batch])
    
    # パディング
    max_len = max(lengths_batch)
    feature_dim = X_batch[0].shape[1]
    batch_size = len(X_batch)
    
    X_padded = torch.zeros(batch_size, max_len, feature_dim)
    for i, x in enumerate(X_batch):
        X_padded[i, :lengths_batch[i], :] = x
    
    return X_padded, y_batch, lengths_batch

# アテンション機構付きLSTMモデル
class AttentionLSTM(nn.Module):
    """
    論文で使用されているアテンション機構付きLSTMモデル
    - 複数層のLSTM
    - アテンション機構（コサイン類似度ベース）
    - BatchNormalization + Dropout
    - 残差接続
    """
    
    def __init__(self, num_layers=2, num_cells=128, num_features_inp=256, 
                 bidir=False, neurons1=128, dropout_fc=0.5, dropout_lstm=0.3):
        super(AttentionLSTM, self).__init__()
        self.num_cells = num_cells
        self.num_features_inp = num_features_inp
        self.bidir = bidir
        self.neurons1 = neurons1
        self.num_layers = num_layers
        self.dropout_fc = dropout_fc
        self.dropout_lstm = dropout_lstm
        
        # LSTM層を構築
        rnns = nn.ModuleList()
        for i in range(self.num_layers):
            input_size_u = self.num_features_inp if i == 0 else self.num_cells
            lstm_layer = nn.LSTM(
                input_size=input_size_u, 
                hidden_size=self.num_cells, 
                num_layers=1, 
                batch_first=True,
                bidirectional=self.bidir
            )
            
            # LSTMのバイアス初期化（forget gateを1に）
            for names in lstm_layer._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(lstm_layer, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)
            
            rnns.append(lstm_layer)
        
        self.rnns = rnns
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 出力サイズの計算
        size = 2 * self.num_cells if self.bidir else self.num_cells
        
        # 全結合層
        self.fc1 = nn.Linear(size, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, 11)
        
        # BatchNormalization
        self.bn1 = nn.BatchNorm1d(size)
        self.bn2 = nn.BatchNorm1d(self.neurons1)
        
        # Dropout
        self.lstm_dropout = nn.Dropout(p=self.dropout_lstm, inplace=True)
        self.fc_dropout = nn.Dropout(p=self.dropout_fc)
    
    def forward(self, X, X_lengths):
        # パディングを隠す
        X = pack_padded_sequence(X, X_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs = []
        
        # 各LSTM層を通過
        for i in range(self.num_layers):
            output, _ = self.rnns[i](X)
            
            # 最後の層以外はドロップアウトを適用
            if i < (self.num_layers - 1):
                output, lens = pad_packed_sequence(output, batch_first=True)
                self.lstm_dropout(output)
                output = pack_padded_sequence(output, lens.cpu(), batch_first=True, enforce_sorted=False)
            
            outputs.append(output)
            X = output
        
        # 最初の層の出力を取得
        X, _ = pad_packed_sequence(outputs[0], batch_first=True)
        
        # 残差接続：各層の出力を加算
        for i in range(1, self.num_layers):
            temp, _ = pad_packed_sequence(outputs[i], batch_first=True)
            X += temp
        
        # アテンション機構の適用
        X = torch.swapaxes(X, 1, 2)
        q = self.pool(X)
        X = torch.swapaxes(X, 1, 2)
        
        # コサイン類似度ベースのアテンション重み計算
        norm_vec = torch.linalg.norm(X, dim=2)
        norm_vec = torch.unsqueeze(norm_vec, dim=2)
        if self.bidir:
            norm_mat = torch.tile(norm_vec, (1, 1, 2*self.num_cells))
        else:
            norm_mat = torch.tile(norm_vec, (1, 1, self.num_cells))
        
        X_uninorm = torch.div(X, norm_mat)
        X_uninorm = torch.nan_to_num(X_uninorm)
        
        wts = torch.bmm(X_uninorm, q)
        wts = F.softmax(wts, dim=1)
        
        # アテンション重みを適用
        X = torch.swapaxes(X, 1, 2)
        X = torch.bmm(X, wts)
        X = torch.reshape(X, (X.shape[0], X.shape[1]))
        
        # 全結合層
        X = self.bn1(X)
        X = self.fc_dropout(F.relu(self.bn2(self.fc1(X))))
        X = self.fc2(X)
        
        return X

# 学習関数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (X_batch, y_batch, lengths_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch, lengths_batch)
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total

# テスト関数
def test_epoch(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch, lengths_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch, lengths_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_batch, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(test_loader), 100 * correct / total

# メイン処理
def main():
    # データの読み込み
    print("\n[1] データの読み込み...")
    data_dir = "./data/"
    train_file = os.path.join(data_dir, "train.pickle")
    test_file = os.path.join(data_dir, "test.pickle")
    
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"  ✓ 学習データのジェスチャー数: {len(train_data)}")
    print(f"  ✓ テストデータのジェスチャー数: {len(test_data)}")
    
    # データの前処理
    print("\n[2] データの前処理...")
    X_train, y_train, lengths_train = get_data(train_data)
    X_test, y_test, lengths_test = get_data(test_data)
    
    print(f"  ✓ 学習サンプル数: {len(X_train)}")
    print(f"  ✓ テストサンプル数: {len(X_test)}")
    print(f"  ✓ 特徴量の次元: {X_train[0].shape[1]}")
    print(f"  ✓ シーケンス長の範囲（学習）: {lengths_train.min()} - {lengths_train.max()}")
    
    # データローダーの作成
    print("\n[3] データローダーの作成...")
    train_dataset = GestureDataset(X_train, y_train, lengths_train)
    test_dataset = GestureDataset(X_test, y_test, lengths_test)
    
    batch_size_train = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"  ✓ 学習バッチ数: {len(train_loader)}")
    print(f"  ✓ テストバッチ数: {len(test_loader)}")
    
    # モデルの作成
    print("\n[4] モデルの作成...")
    model = AttentionLSTM(
        num_layers=2,
        num_cells=128,
        num_features_inp=256,
        bidir=False,
        neurons1=128,
        dropout_fc=0.5,
        dropout_lstm=0.3
    )
    model = model.to(device)
    
    print(f"  ✓ モデル: 二層LSTM + アテンション機構")
    print(f"  ✓ パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 学習の設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    
    print(f"\n[5] 学習開始（{num_epochs}エポック）...")
    print("="*60)
    
    best_test_acc = 0.0
    train_history = []
    test_history = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        train_history.append(train_acc)
        test_history.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # ベストモデルを保存
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
              + (" ★ Best!" if test_acc == best_test_acc else ""))
    
    # 結果の表示
    print("\n" + "="*60)
    print("学習完了！")
    print("="*60)
    print(f"最終学習精度: {train_history[-1]:.2f}%")
    print(f"最終テスト精度: {test_history[-1]:.2f}%")
    print(f"最高テスト精度: {best_test_acc:.2f}% (Epoch {test_history.index(max(test_history))+1})")
    print(f"\n論文の結果と比較:")
    print(f"  論文（二層LSTM）: 99.04%")
    print(f"  今回の実装: {best_test_acc:.2f}%")
    print(f"  差: {99.04 - best_test_acc:.2f}%")
    print("="*60)
    
    # グラフの保存
    print("\n[6] 学習曲線の保存...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_history, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, num_epochs + 1), test_history, 'r-', label='Test Accuracy', linewidth=2)
    plt.axhline(y=99.04, color='g', linestyle='--', label='Paper Result (99.04%)', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Progress - Double Layer LSTM with Attention', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    epochs_zoom = range(10, num_epochs + 1)
    plt.plot(epochs_zoom, train_history[9:], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_zoom, test_history[9:], 'r-', label='Test Accuracy', linewidth=2)
    plt.axhline(y=99.04, color='g', linestyle='--', label='Paper Result (99.04%)', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Progress (Epoch 10-20, Zoomed)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([95, 100])
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("  ✓ グラフを 'training_results.png' に保存しました")
    
    print("\n✓ すべての処理が完了しました！")

if __name__ == "__main__":
    main()

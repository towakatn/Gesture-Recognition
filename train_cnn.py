"""
Modified LeNet CNN for Gesture Recognition using Soli Dataset
Based on the implementation in CNN_Analysis/Modified_LeNet_with_RDI_input.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# シード設定
def set_seed(seed=42):
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
print(f"Using device: {device}")


class ModifiedLeNet(nn.Module):
    """
    Modified LeNet CNN Architecture
    入力: 40チャンネル x 32 x 32 (RDI画像)
    """
    
    def __init__(self, inp_channels=40, num_filters=6, neurons1=216, neurons2=54):
        """
        Args:
            inp_channels: 入力チャンネル数（RDIの場合は40）
            num_filters: 最初の畳み込み層のフィルター数
            neurons1: 最初の全結合層のニューロン数
            neurons2: 2番目の全結合層のニューロン数
        """
        super(ModifiedLeNet, self).__init__()
        self.inp_channels = inp_channels
        self.num_filters = num_filters
        self.neurons1 = neurons1
        self.neurons2 = neurons2
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(inp_channels, num_filters, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全結合層
        # 入力: 32x32 -> conv1 -> 32x32 -> pool1 -> 16x16 -> conv2 -> 12x12 -> pool2 -> 6x6
        # 6x6 x 12フィルター = 432
        self.fc1 = nn.Linear(432, neurons1)
        self.fc2 = nn.Linear(neurons1, neurons2)
        self.fc3 = nn.Linear(neurons2, 11)  # 11クラス
        
    def forward(self, x):
        # Conv1 + Pool1
        x = F.relu(self.conv1(x))  # (batch, 6, 32, 32)
        x = self.pool1(x)           # (batch, 6, 16, 16)
        
        # Conv2 + Pool2
        x = F.relu(self.conv2(x))  # (batch, 12, 12, 12)
        x = self.pool2(x)           # (batch, 12, 6, 6)
        
        # Flatten
        x = x.view(-1, 432)
        
        # 全結合層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x


class SoliDataset(Dataset):
    """
    Soli Dataset用のカスタムデータセット
    """
    def __init__(self, X, y):
        self.data = X
        self.target = y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index])
        y = torch.FloatTensor(self.target[index])
        return x, y


def get_data(data_dict, number_of_frames=40):
    """
    データ辞書からRDI画像を抽出し、前処理を行う
    
    Args:
        data_dict: pickleファイルから読み込んだデータ辞書
        number_of_frames: 各サンプルのフレーム数（固定長に調整）
    
    Returns:
        X: RDI画像のリスト (40 x 32 x 32)
        y: one-hotラベルのリスト
    """
    X = []
    y = []
    keys = np.arange(11)  # 11ジェスチャー
    
    for key in keys:
        gesture_data = data_dict[key]
        for i in range(len(gesture_data)):
            datapt_arr = gesture_data[i]
            
            # 4つのRDIマップを平均化
            datapt = (datapt_arr[0] + datapt_arr[1] + datapt_arr[2] + datapt_arr[3]) / 4
            num_frames = datapt.shape[0]
            
            # フレーム数の調整
            if num_frames < number_of_frames:
                # 足りない場合はゼロパディング
                size = number_of_frames - num_frames
                use = np.zeros((size, 32, 32))
                datapt = np.vstack((datapt, use))
            elif num_frames > number_of_frames:
                # 多い場合はランダムサンプリング
                indices = np.sort(np.random.choice(num_frames, size=number_of_frames, replace=False))
                datapt = datapt[indices, :, :]
            
            # 正規化 (0-1)
            minimum = datapt.min()
            maximum = datapt.max()
            if maximum > minimum:
                datapt = (datapt - minimum) / (maximum - minimum)
            
            X.append(datapt)
            
            # One-hotラベル
            y_temp = np.zeros(11)
            y_temp[key] = 1
            y.append(y_temp)
    
    return X, y


def train_epoch(model, train_loader, criterion, optimizer, device):
    """1エポック分の学習"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total


def test_epoch(model, test_loader, criterion, device):
    """テスト評価"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(test_loader), 100 * correct / total


def main():
    print("="*60)
    print("Modified LeNet CNN for Soli Gesture Recognition")
    print("="*60)
    
    # ステップ1: データの読み込み
    print("\n[Step 1/6] Loading data...")
    with open('./data/train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/test.pickle', 'rb') as f:
        test_data = pickle.load(f)
    print("✓ Data loaded successfully")
    
    # ステップ2: データの前処理
    print("\n[Step 2/6] Processing data...")
    X_train, y_train = get_data(train_data, number_of_frames=40)
    X_test, y_test = get_data(test_data, number_of_frames=40)
    
    # 検証データの分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Image shape: {X_train[0].shape}")
    
    # ステップ3: データセットとデータローダーの作成
    print("\n[Step 3/6] Creating data loaders...")
    train_dataset = SoliDataset(X_train, y_train)
    val_dataset = SoliDataset(X_val, y_val)
    test_dataset = SoliDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("✓ Data loaders created")
    
    # ステップ4: モデルの構築
    print("\n[Step 4/6] Building model...")
    model = ModifiedLeNet(
        inp_channels=40,
        num_filters=6,
        neurons1=216,
        neurons2=54
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 損失関数と最適化手法
    criterion = nn.MSELoss()  # オリジナルの実装に従う
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ステップ5: 学習
    print("\n[Step 5/6] Training model...")
    num_epochs = 30
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc == best_val_acc:
                print(f"  ★ Best validation accuracy!")
    
    # ステップ6: テスト評価
    print("\n[Step 6/6] Evaluating on test set...")
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"\nPaper Results (for comparison):")
    print(f"  RDI CNN: 93.11%")
    print(f"  Motion Profile CNN: 95.67%")
    print("="*60)
    
    # 学習曲線の可視化
    print("\nGenerating training curves...")
    plt.figure(figsize=(14, 5))
    
    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 精度曲線
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.axhline(y=93.11, color='g', linestyle='--', label='Paper (RDI CNN): 93.11%', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_training_results.png', dpi=150, bbox_inches='tight')
    print("✓ Training curves saved as 'cnn_training_results.png'")
    
    # モデルアーキテクチャの詳細
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE DETAILS")
    print("="*60)
    print(model)
    print("="*60)


if __name__ == "__main__":
    main()

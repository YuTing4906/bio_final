import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CNNClassifier(nn.Module):
    """
    用於 DNA 序列分類的卷積神經網路 (CNN) 模型。
    輸入為獨熱編碼的 DNA 序列。
    """
    def __init__(self, seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE, num_classes=2):
        """
        初始化 CNN 分類器。

        Args:
            seq_len (int): 輸入序列的長度。
            vocab_size (int): 序列的詞彙大小 (例如 DNA 為 4，代表 A,C,G,T)。
            num_classes (int): 分類的類別數量 (例如啟動子/非啟動子為 2)。
        """
        super(CNNClassifier, self).__init__()
        # 輸入形狀預期為: (batch_size, vocab_size, seq_len)
        # vocab_size 作為通道數 (in_channels)

        # 第一卷積層
        self.conv1 = nn.Conv1d(in_channels=vocab_size,
                               out_channels=config.CNN_NUM_FILTERS_CONV1, # 濾波器數量
                               kernel_size=config.CNN_KERNEL_SIZE_CONV1,  # 卷積核大小
                               padding=(config.CNN_KERNEL_SIZE_CONV1 - 1) // 2) # "same" padding，保持序列長度不變
        self.bn1 = nn.BatchNorm1d(config.CNN_NUM_FILTERS_CONV1) # 批次歸一化
        self.pool1 = nn.MaxPool1d(kernel_size=config.CNN_POOL_SIZE1) # 最大池化層
        self.dropout1 = nn.Dropout(0.3) # Dropout 層

        # 第二卷積層
        self.conv2 = nn.Conv1d(in_channels=config.CNN_NUM_FILTERS_CONV1, # 輸入通道數等於上一層的輸出通道數
                               out_channels=config.CNN_NUM_FILTERS_CONV2,
                               kernel_size=config.CNN_KERNEL_SIZE_CONV2,
                               padding=(config.CNN_KERNEL_SIZE_CONV2 - 1) // 2) # "same" padding
        self.bn2 = nn.BatchNorm1d(config.CNN_NUM_FILTERS_CONV2)
        self.pool2 = nn.MaxPool1d(kernel_size=config.CNN_POOL_SIZE2)
        self.dropout2 = nn.Dropout(0.3)

        # 計算展平後全連接層的輸入大小
        # 假設卷積層使用 "same" padding，長度變化主要來自池化層
        len_after_pool1 = seq_len // config.CNN_POOL_SIZE1 # 經過第一次池化後的長度
        len_after_pool2 = len_after_pool1 // config.CNN_POOL_SIZE2 # 經過第二次池化後的長度
        self.fc1_input_size = config.CNN_NUM_FILTERS_CONV2 * len_after_pool2 # 全連接層輸入維度

        # 全連接層
        self.fc1 = nn.Linear(self.fc1_input_size, config.CNN_HIDDEN_FC1) # 第一全連接層
        self.dropout_fc = nn.Dropout(0.5) # 全連接層後的 Dropout
        self.fc2 = nn.Linear(config.CNN_HIDDEN_FC1, num_classes) # 第二全連接層 (輸出層)

    def forward(self, x):
        """
        模型的前向傳播。

        Args:
            x (torch.Tensor): 輸入的獨熱編碼序列批次，
                              形狀為 (batch_size, vocab_size, seq_len)。

        Returns:
            torch.Tensor: 每個樣本的分類 logits，形狀為 (batch_size, num_classes)。
        """
        # x 初始形狀: (batch_size, vocab_size, seq_len)

        # 第一卷積塊
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 第二卷積塊
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # 展平操作，為全連接層做準備
        x = x.view(x.size(0), -1) # x.size(0) 是 batch_size

        # 全連接塊
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x) # 輸出 logits (未經 softmax)
        return x

if __name__ == '__main__':
    # 此區塊用於測試 CNN 模型的功能
    print("測試 CNN 分類器模型...")
    device = config.DEVICE
    model = CNNClassifier().to(device)

    # 創建一個虛假的輸入張量進行測試
    # 輸入形狀: (批次大小, 詞彙大小, 序列長度)
    dummy_input = torch.randn(config.CNN_BATCH_SIZE, config.VOCAB_SIZE, config.SEQ_LEN).to(device)
    output = model(dummy_input)
    print(f"CNN 輸出形狀: {output.shape}") # 應為 (batch_size, num_classes)
    print(f"CNN 第一全連接層 (fc1) 的計算輸入大小: {model.fc1_input_size}")
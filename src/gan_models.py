import torch
import torch.nn as nn
import torch.nn.functional as F # 通常在 forward 方法中使用
import config
import numpy as np

class CNNGenerator(nn.Module):
    """
    基於卷積神經網路 (CNN) 的生成器模型 (Generator)。
    用於從潛在向量生成 DNA 序列的獨熱編碼近似 (機率分佈)。
    """
    def __init__(self, latent_dim=config.GAN_LATENT_DIM, seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE):
        """
        初始化 CNN 生成器。

        Args:
            latent_dim (int): 潛在空間的維度。
            seq_len (int): 生成序列的目標長度。
            vocab_size (int): 序列的詞彙大小 (例如 DNA 為 4)。
        """
        super(CNNGenerator, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # 設定初始序列長度，以便後續的反卷積層能擴展到目標 seq_len。
        # 假設通過 3 個 stride=2 的反卷積層，則初始長度約為 seq_len / (2^3) = seq_len / 8。
        # 初始通道數可以設得較大。
        self.init_seq_len = seq_len // 8
        if self.init_seq_len == 0: # 避免因 seq_len 過小導致 init_seq_len 為 0
            self.init_seq_len = 1 # 強制設為 1，但可能影響性能
            print(f"警告: Generator 的 init_seq_len 被設為 1，因為 config.SEQ_LEN ({seq_len}) 太小。")
            print("這可能會影響生成器性能，請考慮增加 SEQ_LEN 或調整生成器架構。")

        # 全連接層，將潛在向量放大並重塑為適合反卷積的初始尺寸
        self.fc_init = nn.Linear(latent_dim, 256 * self.init_seq_len)

        self.model = nn.Sequential(
            # 輸入形狀: (Batch, 256, init_seq_len)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # 輸出: (Batch, 128, init_seq_len*2)
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # 輸出: (Batch, 64, init_seq_len*4)
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, vocab_size, kernel_size=4, stride=2, padding=1), # 輸出: (Batch, vocab_size, init_seq_len*8)
            # 最後一層直接輸出 logits，然後通過 Softmax 轉換為機率分佈
        )

    def forward(self, z):
        """
        生成器前向傳播。

        Args:
            z (torch.Tensor): 潛在向量，形狀為 (batch_size, latent_dim)。

        Returns:
            torch.Tensor: 生成的序列批次，形狀為 (batch_size, vocab_size, seq_len)，
                          值為每個位置上各核苷酸的機率。
        """
        # z 的形狀: (batch_size, latent_dim)
        out = self.fc_init(z)
        out = out.view(out.size(0), 256, self.init_seq_len) # 重塑為 (batch, channels, length)

        out = self.model(out)
        # out 的形狀: (batch_size, vocab_size, target_len_after_convtranspose)

        # 確保輸出長度與目標 seq_len 匹配。
        # ConvTranspose1d 的輸出長度計算公式為: L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
        # 這裡假設經過三次 stride=2 的反卷積後，長度變為 init_seq_len * 8。
        # 若 init_seq_len * 8 不完全等於 seq_len，則進行裁剪或填充。
        current_len = out.shape[2]
        if current_len > self.seq_len:
            out = out[:, :, :self.seq_len] # 裁剪多餘部分
        elif current_len < self.seq_len:
            # 若長度不足，使用 F.pad 填充。更好的做法是調整網絡結構或參數。
            padding_needed = self.seq_len - current_len
            out = F.pad(out, (0, padding_needed)) # 在序列末尾 (右側) 填充
            # print(f"警告: Generator 輸出長度 {current_len} 小於目標 {self.seq_len}，已填充。")

        # 對每個核苷酸位置應用 Softmax，得到機率分佈。
        # 輸出形狀為 (batch_size, vocab_size, seq_len)。
        # GAN 的生成器輸出是否使用 Softmax 取決於評判器的設計和損失函數。
        # 對於 WGAN-GP，評判器直接評估 logits 可能更好。
        # 但若下游任務需要機率，或為了與獨熱編碼的真實數據格式一致，這裡使用 Softmax。
        out_permuted = out.permute(0, 2, 1) # 轉換為 (batch_size, seq_len, vocab_size) 以便在 vocab_size 維度上應用 softmax
        out_softmax = torch.softmax(out_permuted, dim=2) # 在核苷酸類別維度上應用 softmax

        return out_softmax.permute(0, 2, 1) # 轉換回 (batch_size, vocab_size, seq_len)

class Critic(nn.Module):
    """
    WGAN-GP 中的評判器 (Critic) 模型，基於 CNN。
    用於評估輸入序列的真實性，輸出一個無界的評分。
    """
    def __init__(self, seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE):
        """
        初始化 Critic 模型。

        Args:
            seq_len (int): 輸入序列的長度。
            vocab_size (int): 序列的詞彙大小。
        """
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            # 輸入形狀: (batch_size, vocab_size, seq_len)
            nn.Conv1d(vocab_size, 64, kernel_size=7, stride=1, padding=3), # padding="same"
            nn.LeakyReLU(0.2, inplace=True),
            # WGAN-GP 通常不建議在 Critic 中使用 BatchNorm 或 Dropout，
            # 但可以嘗試 InstanceNorm 等其他歸一化方法。
            # nn.InstanceNorm1d(64, affine=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            # 輸出長度計算 (例): L_out = floor((L_in - K + 2P)/S + 1)
            # 若 L_in=251, K=5, P=2, S=2: L_out = floor((251-5+4)/2 + 1) = floor(125+1) = 126
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm1d(128, affine=True),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # L_out = floor(126/2 + 0.5) approx 63
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm1d(256, affine=True),

            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2), # L_out = floor(63/2 + 0.5) approx 32
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm1d(512, affine=True),
        )

        # 動態計算全連接層的輸入大小
        # 創建一個虛擬輸入以推斷卷積部分的輸出形狀
        with torch.no_grad(): # 不需要計算梯度
            dummy_input = torch.zeros(1, vocab_size, seq_len)
            conv_out = self.model(dummy_input)
            self.fc_input_size = int(np.prod(conv_out.size()[1:])) # 展平後的大小
            # print(f"Critic 全連接層輸入大小: {self.fc_input_size}")

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 1)
            # Critic 的最後一層不使用 Sigmoid 激活函數，直接輸出一個評分 (score)。
        )

    def forward(self, seq_one_hot):
        """
        Critic 前向傳播。

        Args:
            seq_one_hot (torch.Tensor): 輸入的序列批次 (獨熱編碼或近似的機率分佈)，
                                       形狀為 (batch_size, vocab_size, seq_len)。

        Returns:
            torch.Tensor: 對每個輸入序列的評分，形狀為 (batch_size, 1)。
        """
        # seq_one_hot 形狀: (batch_size, vocab_size, seq_len)
        conv_out = self.model(seq_one_hot)
        conv_out_flat = conv_out.view(conv_out.size(0), -1) # 將卷積輸出展平
        validity = self.fc(conv_out_flat) # 得到評分
        return validity

if __name__ == '__main__':
    # 此區塊用於測試 GAN 模型的功能
    print("測試 CNN 기반 GAN 模型...")
    device = config.DEVICE

    # 測試 CNN 生成器
    cnn_generator = CNNGenerator().to(device)
    z_cnn = torch.randn(config.GAN_BATCH_SIZE, config.GAN_LATENT_DIM).to(device)
    fake_seqs_cnn = cnn_generator(z_cnn)
    print(f"CNN 生成器輸出形狀: {fake_seqs_cnn.shape}") # 應為 (batch_size, vocab_size, seq_len)
    if fake_seqs_cnn.numel() > 0 and fake_seqs_cnn.shape[2] > 0: # 確保有元素且序列長度大於0
         print(f"CNN 生成器輸出每個位置總和 (應接近 1): {torch.sum(fake_seqs_cnn[0, :, 0])}")

    # 測試 Critic (評判器)
    critic = Critic().to(device)
    score_cnn = critic(fake_seqs_cnn)
    print(f"Critic 輸出形狀 (使用 CNN Generator 的輸出): {score_cnn.shape}") # 應為 (batch_size, 1)
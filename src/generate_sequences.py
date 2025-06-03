import torch
import os
import numpy as np

import config
from gan_models import CNNGenerator 


def generate_synthetic_data(num_samples, generator_path, output_path):
    """
    使用已訓練的 GAN 生成器生成合成的 DNA 序列資料。

    Args:
        num_samples (int): 要生成的合成序列數量。
        generator_path (str): 已訓練的生成器模型檔案路徑 (.pth)。
        output_path (str): 合成資料的儲存路徑 (.pt)。
    """
    print(f"從 {generator_path} 載入已訓練的生成器...")
    device = config.DEVICE

    # 初始化生成器模型結構
    generator = CNNGenerator(latent_dim=config.GAN_LATENT_DIM,
                          seq_len=config.SEQ_LEN,
                          vocab_size=config.VOCAB_SIZE).to(device)

    # 載入已訓練的模型權重
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval() # 設定為評估模式，這會關閉 Dropout 和 BatchNorm 的更新

    print(f"正在生成 {num_samples} 筆合成啟動子序列...")
    all_synthetic_seqs_list = [] # 用於收集所有生成的序列批次

    # 使用 torch.no_grad() 以停用梯度計算，節省記憶體並加速
    with torch.no_grad():
        # 分批生成以避免記憶體不足
        # np.ceil 用於確保即使 num_samples 不是 GAN_BATCH_SIZE 的整倍數也能生成足夠樣本
        for _ in range(int(np.ceil(num_samples / config.GAN_BATCH_SIZE))):
            # 計算當前批次需要生成的樣本數
            num_already_generated = sum(s.shape[0] for s in all_synthetic_seqs_list)
            batch_size_to_generate = min(config.GAN_BATCH_SIZE, num_samples - num_already_generated)

            if batch_size_to_generate <= 0: # 如果已生成足夠樣本，則跳出循環
                break

            # 從標準常態分佈中抽樣潛在向量 z
            z = torch.randn(batch_size_to_generate, config.GAN_LATENT_DIM, device=device)
            # 生成器輸出形狀為 (batch_size, vocab_size, seq_len)
            synthetic_seqs_batch_probs = generator(z)

            # 轉換回 (batch_size, seq_len, vocab_size) 以便與 data_utils 中的格式一致
            # 並移至 CPU 轉換為 NumPy 陣列
            all_synthetic_seqs_list.append(synthetic_seqs_batch_probs.permute(0, 2, 1).cpu().numpy())

    # 將所有批次的序列合併為一個大的 NumPy 陣列
    synthetic_sequences_np = np.concatenate(all_synthetic_seqs_list, axis=0)
    # 確保最終樣本數量準確無誤 (因批次生成可能略多)
    synthetic_sequences_np = synthetic_sequences_np[:num_samples]

    print(f"已生成 {synthetic_sequences_np.shape[0]} 筆合成序列。")
    print(f"合成序列形狀: {synthetic_sequences_np.shape}") # 應為 (num_samples, seq_len, vocab_size)

    # 儲存生成的序列和對應的標籤 (所有標籤都是 1，代表啟動子)
    # 標籤使用 NumPy 的 int64 型別，符合 PyTorch 的 torch.long
    synthetic_labels_np = np.ones(synthetic_sequences_np.shape[0], dtype=np.int64)

    data_to_save = {
        'sequences': torch.tensor(synthetic_sequences_np, dtype=torch.float32), # 序列儲存為 float32 張量
        'labels': torch.tensor(synthetic_labels_np, dtype=torch.long)         # 標籤儲存為 long 張量
    }

    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir: # 確保 output_dir 不是空字串
        os.makedirs(output_dir)

    torch.save(data_to_save, output_path)
    print(f"合成資料已儲存到: {output_path}")

def main():
    """主函數，調用 generate_synthetic_data 生成合成資料。"""
    generate_synthetic_data(
        num_samples=config.NUM_SYNTHETIC_SAMPLES_TO_GENERATE,
        generator_path=config.GAN_SAVE_GENERATOR_PATH,
        output_path=config.SYNTHETIC_DATA_PATH
    )

if __name__ == "__main__":
    main()
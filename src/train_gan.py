import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt

import config
from gan_models import CNNGenerator, Critic # 導入基於 CNN 的生成器和評判器
from data_utils import load_and_preprocess_data # 導入資料處理工具

def plot_gan_losses(g_losses, d_losses, save_path):
    """
    繪製並儲存 GAN 訓練過程中的生成器和評判器損失曲線。

    Args:
        g_losses (list): 生成器的損失歷史記錄。
        d_losses (list): 評判器 (Critic) 的損失歷史記錄。
        save_path (str): 圖片儲存路徑。
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During Training (WGAN-GP)") # 圖表標題
    plt.plot(g_losses, label="Generator Loss") # 生成器損失曲線
    plt.plot(d_losses, label="Critic Loss")    # 評判器損失曲線
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # 確保儲存圖片的目錄存在
    if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close() # 關閉圖表以釋放記憶體
    print(f"WGAN-GP 損失曲線已儲存到: {save_path}")

def compute_gradient_penalty(critic_model, real_samples_tensor, fake_samples_tensor, device_obj):
    """
    計算 WGAN-GP 中的梯度懲罰 (gradient penalty)。
    此懲罰項旨在強制 Critic 函數的梯度範數接近 1，以滿足 Lipschitz 約束。

    Args:
        critic_model (nn.Module): 評判器模型。
        real_samples_tensor (torch.Tensor): 一批真實樣本。
        fake_samples_tensor (torch.Tensor): 一批由生成器生成的假樣本。
        device_obj (torch.device): 計算所用的設備 (CPU 或 GPU)。

    Returns:
        torch.Tensor: 計算得到的梯度懲罰值 (一個純量)。
    """
    # 產生用於插值的隨機加權係數 alpha，形狀為 (batch_size, 1, 1)
    alpha = torch.rand(real_samples_tensor.size(0), 1, 1, device=device_obj)
    # 將 alpha 擴展以匹配樣本的形狀 (batch_size, vocab_size, seq_len)
    alpha = alpha.expand_as(real_samples_tensor)

    # 創建插值樣本：interpolates = alpha * real + (1 - alpha) * fake
    # 這些插值樣本的梯度將被計算
    interpolates = (alpha * real_samples_tensor + ((1 - alpha) * fake_samples_tensor)).requires_grad_(True)
    # Critic 對插值樣本的評分
    d_interpolates = critic_model(interpolates)

    # 準備計算梯度的輸出目標 (通常為與 d_interpolates 同形狀的全1張量)
    grad_outputs_target = torch.ones(d_interpolates.size(), device=device_obj, requires_grad=False)

    # 計算 Critic 輸出相對於插值樣本的梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,         # 梯度的目標純量 (或向量)
        inputs=interpolates,            # 相對於哪個變數計算梯度
        grad_outputs=grad_outputs_target, # 如果 outputs 不是純量，則需要此項
        create_graph=True,       # 創建計算圖，因為梯度懲罰是 Critic 損失的一部分，需要二次求導
        retain_graph=True,       # 保留計算圖，供後續計算使用
        only_inputs=True,        # 只計算相對於 inputs 的梯度
    )[0] # torch.autograd.grad 返回一個元組，取第一個元素

    gradients = gradients.view(gradients.size(0), -1) # 將梯度展平為 (batch_size, -1)
    # 計算梯度懲罰: (||gradient_norm||_2 - 1)^2
    # gradients.norm(2, dim=1) 計算每個樣本梯度的 L2 範數
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    """WGAN-GP 模型訓練主函數。"""
    print(f"使用設備: {config.DEVICE}")
    # 設定隨機種子以確保實驗可重現性
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("載入啟動子資料用於 WGAN-GP 訓練...")
    # WGAN-GP 通常僅使用正樣本 (此處為啟動子序列) 進行訓練
    X_promoters_encoded, _ = load_and_preprocess_data(
        promoter_file_list=config.PROMOTER_FILES, # 僅提供啟動子檔案
        non_promoter_file_list=[],                 # 不載入非啟動子
        data_dir=config.DATA_DIR,
        seq_len=config.SEQ_LEN
    )

    if X_promoters_encoded.shape[0] == 0:
        print("錯誤：沒有載入任何啟動子資料。WGAN-GP 訓練中止。")
        return
    print(f"已載入 {X_promoters_encoded.shape[0]} 筆啟動子序列用於 WGAN-GP 訓練。")

    # 將 NumPy 陣列轉換為 PyTorch 張量，並調整維度以符合 Conv1D (N, C, L)
    # 原始獨熱編碼為 (N, L, C)，需轉換為 (N, C, L)
    promoters_tensor = torch.tensor(X_promoters_encoded, dtype=torch.float32).permute(0, 2, 1)
    promoter_dataset = TensorDataset(promoters_tensor) # 創建 TensorDataset
    # 創建 DataLoader 以進行批次訓練
    dataloader = DataLoader(promoter_dataset, batch_size=config.GAN_BATCH_SIZE, shuffle=True, drop_last=True)

    # 初始化生成器 (Generator) 和評判器 (Critic)
    generator = CNNGenerator(latent_dim=config.GAN_LATENT_DIM,
                         seq_len=config.SEQ_LEN,
                         vocab_size=config.VOCAB_SIZE).to(config.DEVICE)
    critic = Critic(seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE).to(config.DEVICE)

    # 設定優化器：WGAN-GP 論文建議使用 Adam，並調整 betas 參數
    optimizer_G = optim.Adam(generator.parameters(), lr=config.GAN_LR_G, betas=(config.GAN_BETA1, config.GAN_BETA2))
    optimizer_C = optim.Adam(critic.parameters(), lr=config.GAN_LR_D, betas=(config.GAN_BETA1, config.GAN_BETA2))

    # 儲存訓練過程中的損失歷史
    g_losses_history = []
    c_losses_history = []

    print("開始訓練 WGAN-GP...")
    for epoch in range(config.GAN_EPOCHS):
        epoch_c_loss_sum = 0.0 # 記錄該 epoch Critic 的總損失
        epoch_g_loss_sum = 0.0 # 記錄該 epoch Generator 的總損失
        num_batches_in_epoch = len(dataloader)
        num_g_updates_in_epoch = 0 # 記錄該 epoch Generator 的更新次數

        for i, (real_seqs_batch_tuple,) in enumerate(dataloader): # DataLoader 返回的是元組
            real_seqs = real_seqs_batch_tuple.to(config.DEVICE) # 取出真實序列批次並移至設備
            current_batch_size = real_seqs.size(0)

            # --- 步驟 1: 訓練 Critic (評判器) ---
            optimizer_C.zero_grad() # 清除 Critic 的梯度

            # 從潛在空間隨機抽樣 z 向量
            z = torch.randn(current_batch_size, config.GAN_LATENT_DIM, device=config.DEVICE)
            # 使用生成器生成一批假序列
            # 使用 .detach() 使其不參與 Critic 訓練時對 Generator 的梯度計算
            fake_seqs = generator(z).detach()

            # Critic 對真實樣本和生成樣本的評分
            real_validity_scores = critic(real_seqs)
            fake_validity_scores = critic(fake_seqs)

            # 計算梯度懲罰
            gradient_penalty_term = compute_gradient_penalty(critic, real_seqs.data, fake_seqs.data, config.DEVICE)

            # Critic 損失: Wasserstein 距離的近似 + 梯度懲罰項
            # W-Dist_approx = E[Critic(fake)] - E[Critic(real)]
            # Critic 的目標是最大化 E[Critic(real)] - E[Critic(fake)]，
            # 因此最小化 E[Critic(fake)] - E[Critic(real)] + lambda * GP
            c_loss = torch.mean(fake_validity_scores) - torch.mean(real_validity_scores) + config.LAMBDA_GP * gradient_penalty_term

            c_loss.backward() # 反向傳播計算梯度
            optimizer_C.step() # 更新 Critic 的權重

            epoch_c_loss_sum += c_loss.item()

            # --- 步驟 2: 訓練 Generator (生成器) ---
            # 根據 config.K_DISCRIMINATOR_UPDATE (即 n_critic) 的設定，
            # 每 n_critic 次 Critic 迭代後訓練一次 Generator
            if i % config.K_DISCRIMINATOR_UPDATE == 0:
                optimizer_G.zero_grad() # 清除 Generator 的梯度

                # 生成一批新的假序列用於訓練 Generator
                # (或者使用上面已生成的 fake_seqs，但重新生成可確保與最新的 Generator 參數一致)
                gen_z_for_G = torch.randn(current_batch_size, config.GAN_LATENT_DIM, device=config.DEVICE)
                gen_seqs_for_G = generator(gen_z_for_G) # 這次不 detach，因為要更新生成器

                # Generator 的目標是使其生成的假序列被 Critic 評為高分 (即更像真實序列)
                # G_loss = -E[Critic(generated_fake_sequences)]
                g_loss = -torch.mean(critic(gen_seqs_for_G))

                g_loss.backward() # 反向傳播計算梯度
                optimizer_G.step() # 更新 Generator 的權重

                epoch_g_loss_sum += g_loss.item()
                num_g_updates_in_epoch +=1

        # 計算並記錄每個 epoch 的平均損失
        avg_epoch_c_loss = epoch_c_loss_sum / num_batches_in_epoch
        # 避免 Generator 更新次數為零導致除以零錯誤
        avg_epoch_g_loss = epoch_g_loss_sum / num_g_updates_in_epoch if num_g_updates_in_epoch > 0 else 0.0

        print(
            f"[Epoch {epoch+1}/{config.GAN_EPOCHS}] "
            f"[Avg Critic loss: {avg_epoch_c_loss:.4f}] [Avg Generator loss: {avg_epoch_g_loss:.4f}]"
        )
        c_losses_history.append(avg_epoch_c_loss)
        g_losses_history.append(avg_epoch_g_loss)

        # 定期儲存模型 (例如每 20 個 epoch 或在最後一個 epoch)
        if (epoch + 1) % 20 == 0 or epoch == config.GAN_EPOCHS - 1:
            # 確保模型儲存目錄存在
            if not os.path.exists(config.MODELS_DIR):
                os.makedirs(config.MODELS_DIR)

            # 儲存生成器模型
            torch.save(generator.state_dict(), config.GAN_SAVE_GENERATOR_PATH)
            print(f"在 epoch {epoch+1} 儲存生成器模型到 {config.GAN_SAVE_GENERATOR_PATH}")

            # (可選) 儲存 Critic 模型，有助於後續分析或恢復訓練
            critic_save_path = os.path.join(config.MODELS_DIR, "gan_critic.pth")
            torch.save(critic.state_dict(), critic_save_path)
            print(f"在 epoch {epoch+1} 儲存 Critic 模型到 {critic_save_path}")

    print("WGAN-GP 訓練完成。")
    # 繪製並儲存損失曲線
    plot_gan_losses(g_losses_history, c_losses_history, os.path.join(config.RESULTS_DIR, "wgan_gp_loss_curves.png"))

if __name__ == "__main__":
    # 程式執行入口點
    main()
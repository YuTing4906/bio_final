import torch

# --- 主要路徑設定 ---
DATA_DIR = "../data/"  # 存放資料集的目錄路徑
MODELS_DIR = "../models/"  # 存放訓練好的模型的目錄路徑
RESULTS_DIR = "../results/"  # 存放實驗結果與圖表的目錄路徑

# --- 資料相關設定 ---
SEQ_LEN = 251  # DNA 序列的固定長度
# 用於分類的 FASTA 檔案列表
NON_PROMOTER_FILES = ["Mouse_non_nonprom_big.fa", "Mouse_nonprom.fa"]  # 非啟動子序列檔案
PROMOTER_FILES = ["Mouse_non_tata.fa", "Mouse_tata.fa", "Mouse_tata_dbtss.fa"]  # 啟動子序列檔案

# --- 通用訓練設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自動選擇 GPU 或 CPU
RANDOM_SEED = 42  # 隨機種子，用於確保實驗可重現性

# --- GAN 設定 ---
GAN_LATENT_DIM = 100  # GAN 生成器潛在空間的維度
GAN_LR_G = 0.0001      # WGAN-GP 生成器的學習率 (通常較小)
GAN_LR_D = 0.0001      # WGAN-GP 評判器 (Critic) 的學習率 (通常較小)
GAN_BETA1 = 0.5        # Adam 優化器的 beta1 參數 (WGAN-GP 建議為 0 或 0.5)
GAN_BETA2 = 0.9        # Adam 優化器的 beta2 參數 (WGAN-GP 建議為 0.9)
GAN_BATCH_SIZE = 64    # GAN 訓練時的批次大小
GAN_EPOCHS = 500       # GAN 訓練的總週期數 (WGAN-GP 可能需要調整)
GAN_SAVE_GENERATOR_PATH = MODELS_DIR + "gan_generator.pth"  # GAN 生成器模型的儲存路徑
# GAN 訓練時，評判器(Critic)更新次數相對於生成器更新次數的比例 (n_critic)
# WGAN-GP 論文建議 n_critic = 5
K_DISCRIMINATOR_UPDATE = 5  # 評判器相對於生成器的更新頻率

# --- WGAN-GP 特定設定 ---
LAMBDA_GP = 10         # 梯度懲罰 (gradient penalty) 的係數

# --- 合成資料生成設定 ---
NUM_SYNTHETIC_SAMPLES_TO_GENERATE = 10000  # 期望生成的合成啟動子序列數量
SYNTHETIC_DATA_PATH = DATA_DIR + "synthetic_promoters.pt"  # 儲存生成之合成資料的路徑

# --- GAN 驗證設定 ---
# K-mer 比較範圍
KMER_SIZES_TO_VALIDATE = [3, 4, 5, 6]  # 用於驗證的 k-mer 大小列表
GAN_VALIDATION_PLOT_DIR = RESULTS_DIR + "gan_validation/"  # 儲存 GAN 驗證結果圖表的目錄

# --- CNN 分類器設定 ---
CNN_LR = 0.0001  # CNN 分類器的學習率
CNN_BATCH_SIZE = 64  # CNN 分類器訓練時的批次大小
CNN_EPOCHS = 50  # CNN 分類器訓練的總週期數
CNN_MODEL_REAL_DATA_PATH = MODELS_DIR + "cnn_real_data.pth"  # 使用真實資料訓練的 CNN 模型儲存路徑
CNN_MODEL_AUGMENTED_DATA_PATH = MODELS_DIR + "cnn_augmented_data.pth"  # 使用增強資料訓練的 CNN 模型儲存路徑
CLASSIFICATION_REPORT_PATH = RESULTS_DIR + "classification_report.txt"  # 分類結果報告的儲存路徑

# CNN 架構參數 (範例)
CNN_NUM_FILTERS_CONV1 = 128  # 第一卷積層的濾波器數量
CNN_KERNEL_SIZE_CONV1 = 10   # 第一卷積層的核心大小
CNN_POOL_SIZE1 = 3           # 第一池化層的池化大小
CNN_NUM_FILTERS_CONV2 = 64   # 第二卷積層的濾波器數量
CNN_KERNEL_SIZE_CONV2 = 8    # 第二卷積層的核心大小
CNN_POOL_SIZE2 = 2           # 第二池化層的池化大小
CNN_HIDDEN_FC1 = 256         # 全連接層的隱藏單元數量

# --- 序列編碼 ---
# 核苷酸對應的索引 (用於獨熱編碼)
# 'N' 代表未知核苷酸，通常會被特殊處理或忽略
NUCLEOTIDE_MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
VOCAB_SIZE = 4  # 詞彙大小 (A, C, G, T)
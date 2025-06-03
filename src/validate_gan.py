import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import jensenshannon # 用於計算 Jensen-Shannon 散度
from scipy.stats import entropy as scipy_entropy # 用於計算熵 (避免與其他變數衝突)
from sklearn.model_selection import train_test_split # 用於可辨識性評估
from sklearn.metrics import accuracy_score # 用於可辨識性評估
import torch.nn as nn # 用於可辨識性評估的簡單分類器
import torch.optim as optim # 用於可辨識性評估的簡單分類器
from torch.utils.data import Dataset, DataLoader # 用於可辨識性評估的簡單分類器
import seaborn as sns # 匯入 seaborn 用於繪製更美觀的熱圖
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap # UMAP 的標準導入方式


import config
from data_utils import read_fasta, one_hot_decode, one_hot_encode


def calculate_kmer_freqs(sequences_str_list, k):
    """
    計算給定 DNA 序列字串列表中所有 k-mer 的出現頻率。

    Args:
        sequences_str_list (list): DNA 序列字串的列表。
        k (int): k-mer 中的 k 值。

    Returns:
        tuple: (kmer_freqs, total_kmers)
               kmer_freqs (dict): k-mer 到其頻率的映射。
               total_kmers (int): 序列列表中 k-mer 的總數。
    """
    all_kmers = []
    for seq_str in sequences_str_list:
        if len(seq_str) < k: # 跳過長度不足以形成 k-mer 的序列
            continue
        for i in range(len(seq_str) - k + 1):
            all_kmers.append(seq_str[i:i+k])

    if not all_kmers: # 如果沒有提取到任何 k-mer
        return {}, 0

    kmer_counts = Counter(all_kmers) # 計算每種 k-mer 的出現次數
    total_kmers = sum(kmer_counts.values()) # k-mer 總數

    kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freqs, total_kmers

def calculate_gc_content(sequences_str_list):
    """
    計算每個 DNA 序列字串的 GC 含量。

    Args:
        sequences_str_list (list): DNA 序列字串的列表。

    Returns:
        list: 包含每個序列 GC 含量的列表 (浮點數)。
    """
    gc_contents = []
    for seq_str in sequences_str_list:
        if not seq_str: continue # 跳過空序列
        gc_count = seq_str.count('G') + seq_str.count('C')
        gc_contents.append(gc_count / len(seq_str))
    return gc_contents

def plot_kmer_comparison(real_freqs_dict, synthetic_freqs_dict, k, save_path, top_n=30):
    """
    繪製真實數據和合成數據的 k-mer 頻率比較長條圖。
    僅顯示出現頻率總和最高的 top_n 個 k-mer。

    Args:
        real_freqs_dict (dict): 真實序列的 k-mer 頻率。
        synthetic_freqs_dict (dict): 合成序列的 k-mer 頻率。
        k (int): k-mer 的 k 值。
        save_path (str): 圖表儲存路徑。
        top_n (int): 要顯示的 k-mer 數量上限。
    """
    all_kmers_set = set(real_freqs_dict.keys()) | set(synthetic_freqs_dict.keys())

    # 如果 k-mer 種類過多，選擇頻率總和最高的 top_n 個進行繪製
    if len(all_kmers_set) > top_n * 2 and k >= 4 : # 乘以2是個約略值，確保有足夠選擇
        # 計算每個 k-mer 在真實和合成數據中的頻率總和
        temp_freq_sum = {kmer: real_freqs_dict.get(kmer, 0) + synthetic_freqs_dict.get(kmer, 0) for kmer in all_kmers_set}
        # 選擇頻率總和最高的 top_n 個 k-mer
        top_kmers_list = sorted(temp_freq_sum, key=temp_freq_sum.get, reverse=True)[:top_n]
        sorted_kmers_to_plot = sorted(top_kmers_list) # 按字母順序排列選出的 k-mer
    else:
        sorted_kmers_to_plot = sorted(list(all_kmers_set))


    real_vals = [real_freqs_dict.get(kmer, 0) for kmer in sorted_kmers_to_plot]
    synth_vals = [synthetic_freqs_dict.get(kmer, 0) for kmer in sorted_kmers_to_plot]

    x_indices = np.arange(len(sorted_kmers_to_plot))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_kmers_to_plot) * 0.5), 6)) # 動態調整圖表寬度
    ax.bar(x_indices - bar_width/2, real_vals, bar_width, label='Real Promoters')
    ax.bar(x_indices + bar_width/2, synth_vals, bar_width, label='Synthetic Promoters')

    ax.set_ylabel('Frequency')
    ax.set_title(f'{k}-mer Frequency Distribution Comparison (Top {len(sorted_kmers_to_plot)} by summed Freq.)')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(sorted_kmers_to_plot, rotation=90)
    ax.legend()

    fig.tight_layout() # 自動調整子圖參數以給定緊湊的佈局
    plt.savefig(save_path)
    plt.close()
    print(f"{k}-mer 比較圖已儲存到: {save_path}")

def plot_gc_comparison(real_gc_list, synthetic_gc_list, save_path):
    """
    繪製真實數據和合成數據的 GC 含量分佈直方圖。

    Args:
        real_gc_list (list): 真實序列的 GC 含量列表。
        synthetic_gc_list (list): 合成序列的 GC 含量列表。
        save_path (str): 圖表儲存路徑。
    """
    plt.figure(figsize=(10, 6))
    plt.hist(real_gc_list, bins=30, alpha=0.7, label='Real Promoters GC Content', density=True)
    plt.hist(synthetic_gc_list, bins=30, alpha=0.7, label='Synthetic Promoters GC Content', density=True)
    plt.xlabel('GC Content')
    plt.ylabel('Density')
    plt.title('GC Content Distribution Comparison')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"GC 含量比較圖已儲存到: {save_path}")

def calculate_js_divergence(p_freq_dict, q_freq_dict):
    """
    計算兩個 k-mer 頻率分佈之間的 Jensen-Shannon 散度 (JSD)。

    Args:
        p_freq_dict (dict): 第一個 k-mer 頻率分佈。
        q_freq_dict (dict): 第二個 k-mer 頻率分佈。

    Returns:
        float: Jensen-Shannon 散度值。如果任一分佈為空，返回 NaN。
    """
    all_kmers_sorted = sorted(list(set(p_freq_dict.keys()) | set(q_freq_dict.keys())))
    # 根據 all_kmers_sorted 的順序，將頻率字典轉換為 NumPy 陣列 (機率向量)
    p_probs = np.array([p_freq_dict.get(kmer, 0) for kmer in all_kmers_sorted])
    q_probs = np.array([q_freq_dict.get(kmer, 0) for kmer in all_kmers_sorted])

    if len(p_probs) == 0 or len(q_probs) == 0: # 確保機率向量非空
        return float('nan')

    return jensenshannon(p_probs, q_probs, base=2) # 使用底數為 2 的 JSD

# --- 新增的與修改的輔助函數 ---
def calculate_uniqueness(sequences_str_list):
    """
    計算序列字串列表中的唯一序列比例及其數量。

    Args:
        sequences_str_list (list): DNA 序列字串的列表。

    Returns:
        tuple: (uniqueness_ratio, num_unique_sequences)
               uniqueness_ratio (float): 唯一序列的比例。
               num_unique_sequences (int): 唯一序列的數量。
    """
    if not sequences_str_list: # 如果列表為空
        return 0.0, 0
    unique_sequences = set(sequences_str_list)
    uniqueness_ratio = len(unique_sequences) / len(sequences_str_list)
    return uniqueness_ratio, len(unique_sequences)

def calculate_positional_entropy(sequences_one_hot_encoded, seq_len, vocab_size):
    """
    計算獨熱編碼序列在每個位置上的核苷酸分佈熵 (Shannon Entropy)。

    Args:
        sequences_one_hot_encoded (numpy.ndarray): 獨熱編碼的序列批次，
                                                  形狀 (num_samples, seq_len, vocab_size)。
        seq_len (int): 序列長度。
        vocab_size (int): 詞彙大小。

    Returns:
        tuple: (positional_entropy_values, max_entropy)
               positional_entropy_values (numpy.ndarray): 每個位置的熵值陣列。
               max_entropy (float): 理論最大熵值。
    """
    if sequences_one_hot_encoded.shape[0] == 0: # 如果沒有樣本
        return np.array([]), (np.log2(vocab_size) if vocab_size > 0 else 0)

    # 計算每個位置上各核苷酸的平均頻率 (機率)
    # positional_freqs 形狀: (seq_len, vocab_size)
    positional_freqs = np.mean(sequences_one_hot_encoded, axis=0)

    # 計算每個位置的熵
    # scipy_entropy 函數期望輸入的 Pk 總和為 1，positional_freqs[j, :] 即為此類分佈
    positional_entropy_values = np.array([
        scipy_entropy(positional_freqs[j, :], base=2)
        if np.sum(positional_freqs[j, :]) > 0 else 0 # 避免全零導致 NaN
        for j in range(seq_len)
    ])

    # 理論最大熵 (例如，對於 vocab_size=4，最大熵為 log2(4) = 2 bits)
    max_entropy = np.log2(vocab_size) if vocab_size > 0 else 0

    return positional_entropy_values, max_entropy

def plot_positional_entropy_comparison(real_entropy_array, synth_entropy_array, max_entropy_val, seq_len, save_path):
    """
    繪製真實數據和合成數據的每個位置熵比較折線圖。

    Args:
        real_entropy_array (numpy.ndarray): 真實序列每個位置的熵值。
        synth_entropy_array (numpy.ndarray): 合成序列每個位置的熵值。
        max_entropy_val (float): 理論最大熵值。
        seq_len (int): 序列長度。
        save_path (str): 圖表儲存路徑。
    """
    plt.figure(figsize=(15, 7))
    x_positions = np.arange(seq_len)
    plt.plot(x_positions, real_entropy_array, label=f'Real Promoters (Avg: {np.mean(real_entropy_array):.2f})', alpha=0.7)
    plt.plot(x_positions, synth_entropy_array, label=f'Synthetic Promoters (Avg: {np.mean(synth_entropy_array):.2f})', alpha=0.7)
    if max_entropy_val > 0:
        plt.axhline(y=max_entropy_val, color='r', linestyle='--', label=f'Max Entropy ({max_entropy_val:.2f})')

    plt.xlabel('Position in Sequence')
    plt.ylabel('Shannon Entropy (bits)')
    plt.title('Per-Position Nucleotide Entropy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0) # 熵值不為負
    if max_entropy_val > 0:
         plt.ylim(top=max_entropy_val + 0.1) # 在最大熵線條上方留一些空間
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"每個位置熵比較圖已儲存到: {save_path}")

def plot_nucleotide_preference_heatmap(positional_freqs_array, title_str, save_path,
                                     vocab_map_dict=None, seq_len_to_display=None,
                                     vmin=None, vmax=None):
    """
    繪製每個位置的核苷酸偏好性熱圖。

    Args:
        positional_freqs_array (numpy.ndarray): 每個位置上各核苷酸的頻率，
                                                形狀 (seq_len, vocab_size)。
        title_str (str): 圖表標題。
        save_path (str): 圖表儲存路徑。
        vocab_map_dict (dict, optional): 索引到核苷酸的映射 (用於 y 軸標籤)。
        seq_len_to_display (int, optional): 若序列過長，可指定只顯示前 N 個位置。
        vmin (float, optional): 熱圖顏色映射的最小值。
        vmax (float, optional): 熱圖顏色映射的最大值。
    """
    if positional_freqs_array.ndim != 2 or positional_freqs_array.size == 0:
        print(f"警告: positional_freqs_array 維度不正確或為空，跳過繪製熱圖: {title_str}")
        return

    seq_len, vocab_size = positional_freqs_array.shape

    # 決定實際顯示的序列長度
    if seq_len_to_display is None or seq_len_to_display > seq_len:
        seq_len_to_display = seq_len

    # 熱圖數據需要轉置，使核苷酸在 y 軸，位置在 x 軸
    data_to_plot = positional_freqs_array[:seq_len_to_display, :].T # 形狀 (vocab_size, seq_len_to_display)

    # 設定 y 軸刻度標籤
    if vocab_map_dict is None or len(vocab_map_dict) != vocab_size:
        y_tick_labels = [f"Nuc {i}" for i in range(vocab_size)] # 預設標籤
    else:
        y_tick_labels = [vocab_map_dict[i] for i in range(vocab_size)] # 使用映射的核苷酸名稱

    plt.figure(figsize=(max(10, seq_len_to_display * 0.4), max(4, vocab_size * 0.8))) # 動態調整圖表大小
    sns.heatmap(data_to_plot, annot=True, fmt=".2f", cmap="viridis",
                yticklabels=y_tick_labels,
                xticklabels=range(1, seq_len_to_display + 1), # x 軸從 1 開始
                vmin=vmin, vmax=vmax) # 使用傳入的 vmin 和 vmax
    plt.xlabel("Position")
    plt.ylabel("Nucleotide")
    plt.title(title_str)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"核苷酸偏好性熱圖已儲存到: {save_path}")

def plot_dimensionality_reduction(X_reduced_data, labels_array, title_str, save_path):
    """
    繪製降維後的數據點散點圖，用不同顏色和標記區分真實與合成數據。

    Args:
        X_reduced_data (numpy.ndarray): 降維後的 2D 數據點，形狀 (n_samples, 2)。
        labels_array (numpy.ndarray): 每個數據點的標籤 (0 代表真實, 1 代表合成)，形狀 (n_samples,)。
        title_str (str): 圖表標題。
        save_path (str): 圖表儲存路徑。
    """
    plt.figure(figsize=(10, 8))

    # 根據標籤區分真實數據點和合成數據點的索引
    real_indices = (labels_array == 0)
    synth_indices = (labels_array == 1)

    plt.scatter(X_reduced_data[real_indices, 0], X_reduced_data[real_indices, 1],
                label='Real Promoters', alpha=0.7, s=10) # s 控制點的大小
    plt.scatter(X_reduced_data[synth_indices, 0], X_reduced_data[synth_indices, 1],
                label='Synthetic Promoters', alpha=0.7, s=10, marker='x') # 合成數據使用不同標記

    plt.title(title_str)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"{title_str} 圖已儲存到: {save_path}")

def perform_and_plot_reductions(real_encoded_flat_array, synthetic_encoded_flat_array,
                                plot_directory, num_samples_to_plot=2000):
    """
    對真實和合成序列的獨熱編碼（扁平化後）執行 PCA, t-SNE, UMAP 降維並繪製結果。

    Args:
        real_encoded_flat_array (numpy.ndarray): 扁平化的真實獨熱編碼序列,
                                                 形狀 (n_real_samples, seq_len * vocab_size)。
        synthetic_encoded_flat_array (numpy.ndarray): 扁平化的合成獨熱編碼序列,
                                                      形狀 (n_synth_samples, seq_len * vocab_size)。
        plot_directory (str): 儲存降維圖表的目錄。
        num_samples_to_plot (int): 從真實和合成數據中各抽樣多少樣本進行繪圖，以加速計算。
    """
    print("\n--- 開始執行降維視覺化分析 ---")

    # 抽樣數據以加速計算，尤其是 t-SNE 和 UMAP，它們對大數據集較敏感
    if real_encoded_flat_array.shape[0] > num_samples_to_plot:
        idx_real = np.random.choice(real_encoded_flat_array.shape[0], num_samples_to_plot, replace=False)
        real_sampled_flat = real_encoded_flat_array[idx_real, :]
    else:
        real_sampled_flat = real_encoded_flat_array

    if synthetic_encoded_flat_array.shape[0] > num_samples_to_plot:
        idx_synth = np.random.choice(synthetic_encoded_flat_array.shape[0], num_samples_to_plot, replace=False)
        synthetic_sampled_flat = synthetic_encoded_flat_array[idx_synth, :]
    else:
        synthetic_sampled_flat = synthetic_encoded_flat_array

    if real_sampled_flat.shape[0] == 0 or synthetic_sampled_flat.shape[0] == 0:
        print("  警告: 真實或合成抽樣數據為空，跳過降維視覺化。")
        return

    print(f"  將使用 {real_sampled_flat.shape[0]} 筆真實樣本和 {synthetic_sampled_flat.shape[0]} 筆合成樣本進行降維視覺化。")

    # 合併數據並創建標籤 (0 代表真實, 1 代表合成)
    X_combined_flat = np.vstack((real_sampled_flat, synthetic_sampled_flat))
    labels_combined_array = np.array([0] * real_sampled_flat.shape[0] + [1] * synthetic_sampled_flat.shape[0])

    # 1. PCA 降維
    print("  執行 PCA...")
    try:
        pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
        X_pca_reduced = pca.fit_transform(X_combined_flat)
        plot_dimensionality_reduction(X_pca_reduced, labels_combined_array, "PCA of Real vs. Synthetic Promoters",
                                      os.path.join(plot_directory, "pca_comparison.png"))
    except Exception as e:
        print(f"  PCA 執行失敗: {e}")

    # 2. t-SNE 降維
    print("  執行 t-SNE (可能需要一些時間)...")
    try:
        # t-SNE 的 perplexity 參數通常建議在 5 到 50 之間，並且必須小於樣本數
        perplexity_val = min(30, X_combined_flat.shape[0] - 1) # 確保 perplexity < n_samples
        if perplexity_val < 5 and X_combined_flat.shape[0] > 1: # 如果樣本太少，t-SNE 可能效果不佳或報錯
             print(f"  樣本數 ({X_combined_flat.shape[0]}) 相對較少，已調整 t-SNE perplexity 為 {perplexity_val}")

        if X_combined_flat.shape[0] > 1: # 至少需要2個樣本才能執行 t-SNE
            tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED,
                        perplexity=max(5, perplexity_val), # 確保 perplexity 至少為 5
                        n_iter=300, init='pca') # 使用 PCA 初始化可以加速收斂並提高穩定性
            X_tsne_reduced = tsne.fit_transform(X_combined_flat)
            plot_dimensionality_reduction(X_tsne_reduced, labels_combined_array, "t-SNE of Real vs. Synthetic Promoters",
                                        os.path.join(plot_directory, "tsne_comparison.png"))
        else:
            print("  樣本數不足 (<=1)，無法執行 t-SNE。")

    except Exception as e:
        print(f"  t-SNE 執行失敗: {e}")

    # 3. UMAP 降維
    print("  執行 UMAP...")
    try:
        if X_combined_flat.shape[0] > 1: # 至少需要2個樣本
            # UMAP 的 n_neighbors 參數通常在 5 到 50 之間，且應小於樣本數
            n_neighbors_val = min(15, X_combined_flat.shape[0] - 1)
            # 確保 n_neighbors 至少為 2 (UMAP 要求)
            if n_neighbors_val < 2 and X_combined_flat.shape[0] > 1 :
                n_neighbors_val = X_combined_flat.shape[0] -1
            elif X_combined_flat.shape[0] <= 1: # 若樣本數不足，設為1以避免錯誤，但 UMAP 可能無法正常運行
                n_neighbors_val = 1

            if n_neighbors_val >= 2: # 僅在 n_neighbors 有效時執行
                reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED,
                                    n_neighbors=n_neighbors_val, min_dist=0.1)
                X_umap_reduced = reducer.fit_transform(X_combined_flat)
                plot_dimensionality_reduction(X_umap_reduced, labels_combined_array, "UMAP of Real vs. Synthetic Promoters",
                                            os.path.join(plot_directory, "umap_comparison.png"))
            else:
                 print(f"  樣本數 ({X_combined_flat.shape[0]}) 不足以設定有效的 UMAP n_neighbors (目前為 {n_neighbors_val})。跳過 UMAP。")
        else:
            print("  樣本數不足 (<=1)，無法執行 UMAP。")
    except Exception as e:
        print(f"  UMAP 執行失敗: {e}")

    print("--- 降維視覺化分析完成 ---")


# --- 用於可辨識性評估的簡單 CNN 模型和 Dataset ---
class SimpleCNNForDiscriminability(nn.Module):
    """一個簡單的 CNN 分類器，用於評估真實序列和合成序列之間的可辨識性。"""
    def __init__(self, seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE):
        super(SimpleCNNForDiscriminability, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=32, kernel_size=5, padding=2) # padding="same"
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2) # padding="same"
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 動態計算全連接層的輸入大小
        len_after_pool1 = seq_len // 2
        len_after_pool2 = len_after_pool1 // 2
        self.fc1_input_size = 64 * len_after_pool2

        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 1) # 二元分類 (真實 vs 合成)，輸出 logits

    def forward(self, x):
        # x 輸入形狀: (batch_size, vocab_size, seq_len)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) # 直接輸出 logits，BCEWithLogitsLoss 會處理 sigmoid
        return x

class DiscriminabilityDataset(Dataset):
    """用於可辨識性評估的 PyTorch Dataset。"""
    def __init__(self, sequences_one_hot_encoded, labels_array):
        # sequences_one_hot_encoded 應為 (num_samples, seq_len, vocab_size)
        # PyTorch Conv1D 期望 (N, C, L)，其中 C=vocab_size, L=seq_len
        self.sequences = torch.tensor(sequences_one_hot_encoded, dtype=torch.float32).permute(0, 2, 1)
        self.labels = torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1) # 增加一個維度以匹配 BCEWithLogitsLoss 的期望

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def train_and_evaluate_discriminability(real_encoded_array, synthetic_encoded_array,
                                        seq_len, vocab_size, device, epochs=10):
    """
    訓練一個簡單的 CNN 分類器來區分真實序列和合成序列，並評估其準確率。
    準確率越接近 0.5，表示真實序列和合成序列越難以區分，GAN 的性能可能越好。

    Args:
        real_encoded_array (numpy.ndarray): 真實序列的獨熱編碼。
        synthetic_encoded_array (numpy.ndarray): 合成序列的獨熱編碼。
        seq_len (int): 序列長度。
        vocab_size (int): 詞彙大小。
        device (torch.device): 計算設備。
        epochs (int): 訓練週期數。

    Returns:
        float: 分類器在測試集上的準確率。
    """
    print("  開始訓練可辨識性分類器...")

    # 準備資料: 0 代表真實序列, 1 代表合成序列
    labels_real = np.zeros(real_encoded_array.shape[0])
    labels_synthetic = np.ones(synthetic_encoded_array.shape[0])

    all_sequences_encoded = np.concatenate((real_encoded_array, synthetic_encoded_array), axis=0)
    all_labels = np.concatenate((labels_real, labels_synthetic), axis=0)

    # 打亂資料順序
    indices = np.arange(all_sequences_encoded.shape[0])
    np.random.shuffle(indices)
    all_sequences_encoded = all_sequences_encoded[indices]
    all_labels = all_labels[indices]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences_encoded, all_labels, test_size=0.2, random_state=config.RANDOM_SEED, stratify=all_labels
    )

    train_dataset = DiscriminabilityDataset(X_train, y_train)
    test_dataset = DiscriminabilityDataset(X_test, y_test)

    # 使用較小的批次大小，因為這只是個輔助評估工具
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleCNNForDiscriminability(seq_len=seq_len, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() # 適用於二元分類，且模型最後輸出 logits

    for epoch in range(epochs):
        model.train()
        for seqs, labs in train_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(seqs) # 輸出 logits
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
        # print(f"    可辨識性分類器 Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    all_preds_probs_list = []
    all_true_labels_list = []
    with torch.no_grad():
        for seqs, labs in test_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            outputs_logits = model(seqs)
            all_preds_probs_list.extend(torch.sigmoid(outputs_logits).cpu().numpy()) # 將 logits 轉換為機率
            all_true_labels_list.extend(labs.cpu().numpy())

    # 根據機率閾值 0.5 轉換為預測類別
    all_predicted_labels = (np.array(all_preds_probs_list) > 0.5).astype(int)
    accuracy = accuracy_score(all_true_labels_list, all_predicted_labels)

    print(f"  可辨識性分類器測試準確率: {accuracy:.4f} (越接近 0.5 表示越難區分)")
    return accuracy


def main():
    """GAN 生成資料品質驗證主函數。"""
    print("開始 GAN 生成資料品質驗證...")
    # 確保儲存驗證結果圖表的目錄存在
    if not os.path.exists(config.GAN_VALIDATION_PLOT_DIR):
        os.makedirs(config.GAN_VALIDATION_PLOT_DIR)

    # 1. 載入真實啟動子序列 (同時獲取字串格式和獨熱編碼格式)
    print("載入真實啟動子序列...")
    real_promoter_sequences_str_list = [] # 儲存字串格式的真實序列
    temp_real_promoter_encoded_list = []  # 暫時儲存獨熱編碼的真實序列

    for fname in config.PROMOTER_FILES:
        fpath = os.path.join(config.DATA_DIR, fname)
        seqs_str = read_fasta(fpath) # 讀取 FASTA 檔案
        for s_str in seqs_str:
            if len(s_str) == config.SEQ_LEN: # 只處理符合設定長度的序列
                real_promoter_sequences_str_list.append(s_str)
                temp_real_promoter_encoded_list.append(one_hot_encode(s_str,
                                                                     seq_len=config.SEQ_LEN,
                                                                     mapping=config.NUCLEOTIDE_MAPPING,
                                                                     vocab_size=config.VOCAB_SIZE))
    if not real_promoter_sequences_str_list:
        print("錯誤：未能載入任何符合長度要求的真實啟動子序列。驗證中止。")
        return

    real_promoter_encoded_np_array = np.array(temp_real_promoter_encoded_list)
    print(f"已載入 {len(real_promoter_sequences_str_list)} 筆真實啟動子序列 (字串與獨熱編碼格式)。")

    # 2. 載入並解碼合成啟動子序列 (獲取獨熱編碼格式和字串格式)
    print("載入並解碼合成啟動子序列...")
    if not os.path.exists(config.SYNTHETIC_DATA_PATH):
        print(f"錯誤：找不到合成資料檔案 {config.SYNTHETIC_DATA_PATH}。請先執行 generate_sequences.py。驗證中止。")
        return

    synthetic_data_dict = torch.load(config.SYNTHETIC_DATA_PATH, map_location='cpu') # 確保在 CPU 上載入
    # 合成資料儲存時為 (num_samples, seq_len, vocab_size)
    X_synthetic_encoded_np_array = synthetic_data_dict['sequences'].numpy()

    num_synthetic_to_validate = len(X_synthetic_encoded_np_array)
    print(f"將使用 {num_synthetic_to_validate} 筆合成序列進行驗證。")

    # 將獨熱編碼的合成序列解碼回字串格式
    synthetic_promoter_sequences_str_list = one_hot_decode(X_synthetic_encoded_np_array, config.NUCLEOTIDE_MAPPING)

    if not synthetic_promoter_sequences_str_list:
        print("錯誤：未能解碼任何合成啟動子序列。驗證中止。")
        return

    # 用於儲存所有驗證指標的字典
    validation_summary_dict = {}

    # 3. k-mer 頻率分析與比較
    print("\n--- k-mer 頻率分析 ---")
    all_jsd_results_dict = {}
    for k_val in config.KMER_SIZES_TO_VALIDATE:
        print(f"  分析 {k_val}-mers...")
        real_kmer_freqs, total_real_kmers = calculate_kmer_freqs(real_promoter_sequences_str_list, k_val)
        synth_kmer_freqs, total_synth_kmers = calculate_kmer_freqs(synthetic_promoter_sequences_str_list, k_val)

        if total_real_kmers == 0 or total_synth_kmers == 0:
            print(f"  警告: 對於 {k_val}-mers，真實或合成序列的 k-mer 總數為0，跳過比較。")
            all_jsd_results_dict[f'{k_val}-mer JSD'] = float('nan')
            continue

        jsd_val = calculate_js_divergence(real_kmer_freqs, synth_kmer_freqs)
        all_jsd_results_dict[f'{k_val}-mer JSD'] = jsd_val
        print(f"    {k_val}-mer Jensen-Shannon Divergence: {jsd_val:.6f} (越小越相似)")

        plot_kmer_comparison(real_kmer_freqs, synth_kmer_freqs, k_val,
                             os.path.join(config.GAN_VALIDATION_PLOT_DIR, f"kmer_{k_val}_comparison.png"))
    validation_summary_dict.update(all_jsd_results_dict)

    # 4. GC 含量分析與比較
    print("\n--- GC 含量分析 ---")
    real_gc_list = calculate_gc_content(real_promoter_sequences_str_list)
    synthetic_gc_list = calculate_gc_content(synthetic_promoter_sequences_str_list)

    if not real_gc_list or not synthetic_gc_list:
        print("  警告: 真實或合成序列的 GC 含量列表為空，跳過 GC 比較。")
        validation_summary_dict['Real Avg GC'] = float('nan')
        validation_summary_dict['Synth Avg GC'] = float('nan')
    else:
        mean_real_gc = np.mean(real_gc_list)
        std_real_gc = np.std(real_gc_list)
        mean_synth_gc = np.mean(synthetic_gc_list)
        std_synth_gc = np.std(synthetic_gc_list)
        validation_summary_dict['Real Avg GC'] = mean_real_gc
        validation_summary_dict['Synth Avg GC'] = mean_synth_gc
        print(f"  真實啟動子平均 GC 含量: {mean_real_gc:.4f} (標準差: {std_real_gc:.4f})")
        print(f"  合成啟動子平均 GC 含量: {mean_synth_gc:.4f} (標準差: {std_synth_gc:.4f})")
        plot_gc_comparison(real_gc_list, synthetic_gc_list,
                           os.path.join(config.GAN_VALIDATION_PLOT_DIR, "gc_content_comparison.png"))

    # 5. 唯一序列比例分析
    print("\n--- 唯一序列比例分析 ---")
    real_uniqueness_ratio, num_unique_real = calculate_uniqueness(real_promoter_sequences_str_list)
    synth_uniqueness_ratio, num_unique_synth = calculate_uniqueness(synthetic_promoter_sequences_str_list)
    validation_summary_dict['Real Uniqueness Ratio'] = real_uniqueness_ratio
    validation_summary_dict['Synth Uniqueness Ratio'] = synth_uniqueness_ratio
    print(f"  真實啟動子: {num_unique_real}/{len(real_promoter_sequences_str_list)} 唯一序列 (比例: {real_uniqueness_ratio:.4f})")
    print(f"  合成啟動子: {num_unique_synth}/{len(synthetic_promoter_sequences_str_list)} 唯一序列 (比例: {synth_uniqueness_ratio:.4f})")

    # 6. 每個位置的核苷酸組成與熵分析
    print("\n--- 每個位置的核苷酸組成與熵分析 ---")
    # 計算全局 vmin 和 vmax 以便熱圖顏色範圍一致
    global_vmin, global_vmax = None, None

    if real_promoter_encoded_np_array.shape[0] > 0:
        # real_pos_freqs 形狀: (seq_len, vocab_size)
        real_pos_freqs_array = np.mean(real_promoter_encoded_np_array, axis=0)
        if real_pos_freqs_array.size > 0:
            current_min_real = np.min(real_pos_freqs_array)
            current_max_real = np.max(real_pos_freqs_array)
            global_vmin = current_min_real
            global_vmax = current_max_real
    else:
        real_pos_freqs_array = np.array([]) # 保持為空陣列以便後續檢查

    if X_synthetic_encoded_np_array.shape[0] > 0:
        # synth_pos_freqs 形狀: (seq_len, vocab_size)
        synth_pos_freqs_array = np.mean(X_synthetic_encoded_np_array, axis=0)
        if synth_pos_freqs_array.size > 0:
            current_min_synth = np.min(synth_pos_freqs_array)
            current_max_synth = np.max(synth_pos_freqs_array)
            global_vmin = min(global_vmin, current_min_synth) if global_vmin is not None else current_min_synth
            global_vmax = max(global_vmax, current_max_synth) if global_vmax is not None else current_max_synth
    else:
        synth_pos_freqs_array = np.array([])

    print(f"  計算得到的全局顏色範圍 (用於熱圖): vmin={global_vmin}, vmax={global_vmax}")

    # 獲取核苷酸索引到名稱的映射 (用於熱圖 y 軸標籤)
    nucleotide_idx_to_name_map = {v: k for k, v in config.NUCLEOTIDE_MAPPING.items() if k != 'N'}

    # 繪製核苷酸偏好性熱圖 (使用全局 vmin 和 vmax)
    if real_pos_freqs_array.size > 0:
        plot_nucleotide_preference_heatmap(
            real_pos_freqs_array,
            "Real Promoters Positional Nucleotide Preference",
            os.path.join(config.GAN_VALIDATION_PLOT_DIR, "real_nucleotide_preference.png"),
            vocab_map_dict=nucleotide_idx_to_name_map,
            seq_len_to_display=min(50, config.SEQ_LEN), # 最多顯示前 50bp 或全部
            vmin=global_vmin, vmax=global_vmax
        )
    if synth_pos_freqs_array.size > 0:
        plot_nucleotide_preference_heatmap(
            synth_pos_freqs_array,
            "Synthetic Promoters Positional Nucleotide Preference",
            os.path.join(config.GAN_VALIDATION_PLOT_DIR, "synthetic_nucleotide_preference.png"),
            vocab_map_dict=nucleotide_idx_to_name_map,
            seq_len_to_display=min(50, config.SEQ_LEN),
            vmin=global_vmin, vmax=global_vmax
        )

    # 計算並繪製每個位置的熵
    real_pos_entropy_array, max_entropy_val = calculate_positional_entropy(
        real_promoter_encoded_np_array, config.SEQ_LEN, config.VOCAB_SIZE
    )
    synth_pos_entropy_array, _ = calculate_positional_entropy(
        X_synthetic_encoded_np_array, config.SEQ_LEN, config.VOCAB_SIZE
    )

    if real_pos_entropy_array.size > 0 and synth_pos_entropy_array.size > 0:
        validation_summary_dict['Real Avg Positional Entropy'] = np.mean(real_pos_entropy_array)
        validation_summary_dict['Synth Avg Positional Entropy'] = np.mean(synth_pos_entropy_array)
        print(f"  真實序列平均每個位置熵: {np.mean(real_pos_entropy_array):.4f} bits (最大可能熵: {max_entropy_val:.4f} bits)")
        print(f"  合成序列平均每個位置熵: {np.mean(synth_pos_entropy_array):.4f} bits")
        plot_positional_entropy_comparison(real_pos_entropy_array, synth_pos_entropy_array, max_entropy_val, config.SEQ_LEN,
                                           os.path.join(config.GAN_VALIDATION_PLOT_DIR, "positional_entropy_comparison.png"))
    else:
        print("  警告: 無法計算每個位置的熵 (可能真實或合成數據為空)。")
        validation_summary_dict['Real Avg Positional Entropy'] = float('nan')
        validation_summary_dict['Synth Avg Positional Entropy'] = float('nan')

    # 7. 可辨識性評估 (真實 vs. 合成)
    print("\n--- 可辨識性評估 (真實 vs. 合成) ---")
    device = config.DEVICE
    # 使用相同數量的真實和合成樣本進行評估，以避免類別不平衡影響
    num_samples_for_disc_eval = min(len(real_promoter_encoded_np_array), len(X_synthetic_encoded_np_array))

    if num_samples_for_disc_eval > 10: # 確保有足夠樣本進行訓練和測試 (至少 > batch_size/test_split_ratio)
        # 從真實和合成數據中隨機抽取相同數量的樣本
        idx_real_sample = np.random.choice(len(real_promoter_encoded_np_array), num_samples_for_disc_eval, replace=False)
        idx_synth_sample = np.random.choice(len(X_synthetic_encoded_np_array), num_samples_for_disc_eval, replace=False)

        discriminability_accuracy = train_and_evaluate_discriminability(
            real_promoter_encoded_np_array[idx_real_sample],
            X_synthetic_encoded_np_array[idx_synth_sample],
            config.SEQ_LEN,
            config.VOCAB_SIZE,
            device
        )
        validation_summary_dict['Discriminability Score (Accuracy)'] = discriminability_accuracy
    else:
        print(f"  警告: 真實或合成樣本過少 (各 {num_samples_for_disc_eval} 筆)，跳過可辨識性評估。")
        validation_summary_dict['Discriminability Score (Accuracy)'] = float('nan')

    # 8. 降維視覺化分析
    # 準備扁平化的獨熱編碼數據 (將 seq_len 和 vocab_size 維度合併)
    if real_promoter_encoded_np_array.size > 0 and X_synthetic_encoded_np_array.size > 0:
        real_flat_array = real_promoter_encoded_np_array.reshape(real_promoter_encoded_np_array.shape[0], -1)
        synth_flat_array = X_synthetic_encoded_np_array.reshape(X_synthetic_encoded_np_array.shape[0], -1)

        perform_and_plot_reductions(real_flat_array, synth_flat_array,
                                    config.GAN_VALIDATION_PLOT_DIR,
                                    num_samples_to_plot=min(2000, num_samples_for_disc_eval)) # 使用較小樣本數進行繪圖
    else:
        print("警告: 真實或合成獨熱編碼數據為空，無法執行降維視覺化。")

    # 9. 輸出量化結果總結
    print("\n--- GAN 品質驗證量化總結 ---")
    for metric, value in validation_summary_dict.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    summary_file_path = os.path.join(config.RESULTS_DIR, "gan_validation_summary.txt")
    with open(summary_file_path, 'w', encoding='utf-8') as f: # 指定 UTF-8 編碼
        f.write("GAN 品質驗證量化總結\n")
        f.write("="*30 + "\n")
        for metric, value in validation_summary_dict.items():
            if isinstance(value, float):
                f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write(f"  {metric}: {value}\n")
    print(f"\n量化總結已儲存到: {summary_file_path}")

    print("\nGAN 品質驗證完成。圖表已儲存到:", config.GAN_VALIDATION_PLOT_DIR)

if __name__ == "__main__":
    # 簡易測試設定 (如果直接執行此檔案進行測試)
    # 確保目錄存在
    if not os.path.exists(config.DATA_DIR): os.makedirs(config.DATA_DIR)
    if not os.path.exists(config.MODELS_DIR): os.makedirs(config.MODELS_DIR)
    if not os.path.exists(config.RESULTS_DIR): os.makedirs(config.RESULTS_DIR)
    if not os.path.exists(config.GAN_VALIDATION_PLOT_DIR): os.makedirs(config.GAN_VALIDATION_PLOT_DIR)

    main()
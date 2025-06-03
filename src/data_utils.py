import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import config

def read_fasta(file_path):
    """
    從 FASTA 格式的檔案中讀取 DNA 序列。

    Args:
        file_path (str): FASTA 檔案的路徑。

    Returns:
        list: 包含所有序列的列表，序列均轉換為大寫。
    """
    sequences = []
    with open(file_path, 'r') as f:
        sequence = ""
        for line in f:
            if line.startswith('>'):  # FASTA 格式的序列標頭行
                if sequence:  # 如果 sequence 非空，表示上一個序列已讀取完畢
                    sequences.append(sequence.upper())
                sequence = ""  # 開始讀取新的序列
            else:
                sequence += line.strip()  # 移除換行符並串接序列片段
        if sequence:  # 加入檔案中的最後一個序列
            sequences.append(sequence.upper())
    return sequences

def one_hot_encode(sequence, seq_len=config.SEQ_LEN, mapping=config.NUCLEOTIDE_MAPPING, vocab_size=config.VOCAB_SIZE):
    """
    將 DNA 序列轉換為獨熱編碼 (one-hot encoding) 的 NumPy 陣列。

    Args:
        sequence (str): 輸入的 DNA 序列字串。
        seq_len (int, optional): 預期的序列長度。預設為 config.SEQ_LEN。
                                 序列會被截斷或（若要實現）填充至此長度。
        mapping (dict, optional): 核苷酸到索引的映射。預設為 config.NUCLEOTIDE_MAPPING。
        vocab_size (int, optional): 詞彙大小 (即核苷酸種類數)。預設為 config.VOCAB_SIZE。

    Returns:
        numpy.ndarray: 獨熱編碼後的序列，形狀為 (seq_len, vocab_size)。
                       例如，'A' 可能被編碼為 [1,0,0,0]。
                       未知核苷酸 ('N' 或其他) 將保持為全零向量。
    """
    arr = np.zeros((seq_len, vocab_size), dtype=np.float32)
    # 假設所有序列長度已符合 seq_len，或在此處進行截斷
    for i, nucleotide in enumerate(sequence[:seq_len]):
        if nucleotide in mapping and mapping[nucleotide] < vocab_size:  # 確保核苷酸有效且索引在範圍內
            arr[i, mapping[nucleotide]] = 1.0
        # 'N' 或其他未定義的核苷酸將保持為 [0,0,0,0] (隱式處理)
        # 也可以考慮將 'N' 編碼為 [0.25, 0.25, 0.25, 0.25] 等方式，視需求而定。
    # PyTorch 的 Conv1D 期望輸入形狀為 (Channels, Length)。
    # 此函數返回 (Length, Channels)，後續在 Dataset 或模型中需進行維度轉換。
    return arr  # 形狀為 (seq_len, vocab_size)

def one_hot_decode(encoded_sequence_batch, mapping=config.NUCLEOTIDE_MAPPING):
    """
    將一批獨熱編碼的序列 (或機率分佈) 轉換回 DNA 字串列表。

    Args:
        encoded_sequence_batch (numpy.ndarray): 獨熱編碼的序列批次。
                                                可以是單個序列 (seq_len, vocab_size)
                                                或一批序列 (batch_size, seq_len, vocab_size)。
        mapping (dict, optional): 核苷酸到索引的映射。預設為 config.NUCLEOTIDE_MAPPING。

    Returns:
        list: 解碼後的 DNA 字串列表。
    """
    if encoded_sequence_batch.ndim == 2:  # 若輸入為單個序列，擴展維度以統一處理
        encoded_sequence_batch = np.expand_dims(encoded_sequence_batch, axis=0)

    # 建立反向映射 (索引 -> 核苷酸)，排除 'N' (因為 'N' 在編碼時是全零)
    inv_mapping = {v: k for k, v in mapping.items() if k != 'N'}
    dna_strings = []

    for i in range(encoded_sequence_batch.shape[0]):  # 遍歷批次中的每個序列
        seq_encoded = encoded_sequence_batch[i]  # 當前序列，形狀為 (seq_len, vocab_size)
        dna_string = ""
        for j in range(seq_encoded.shape[0]):  # 遍歷序列中的每個位置
            # 從機率分佈中選擇機率最高的核苷酸索引。
            # 如果是嚴格的獨熱編碼，argmax 也能正確工作。
            nucleotide_idx = np.argmax(seq_encoded[j])
            dna_string += inv_mapping.get(nucleotide_idx, '?')  # 若索引未知，則用 '?' 表示
        dna_strings.append(dna_string)
    return dna_strings

class DNASequenceDataset(Dataset):
    """
    用於 DNA 序列的 PyTorch Dataset 類別。
    假設輸入的序列已經是獨熱編碼格式。
    """
    def __init__(self, sequences, labels, seq_len=config.SEQ_LEN):
        """
        初始化 Dataset。

        Args:
            sequences (numpy.ndarray): 獨熱編碼後的序列數據，形狀應為 (num_samples, seq_len, vocab_size)。
            labels (numpy.ndarray): 對應的標籤數據。
            seq_len (int, optional): 序列長度。預設為 config.SEQ_LEN。
        """
        self.sequences = sequences  # 預期為 (num_samples, seq_len, vocab_size)
        self.labels = labels
        self.seq_len = seq_len # 此處 seq_len 參數可能未使用，因為序列已是固定長度

    def __len__(self):
        """返回數據集的樣本總數。"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        獲取指定索引的樣本。

        Args:
            idx (int): 樣本索引。

        Returns:
            tuple: (sequence_tensor, label_tensor)
                   sequence_tensor 形狀為 (vocab_size, seq_len)，符合 PyTorch Conv1D 輸入。
                   label_tensor 為長整型標籤。
        """
        sequence = self.sequences[idx]  # 形狀應為 (seq_len, vocab_size)
        label = self.labels[idx]

        # 轉換為 PyTorch 張量，並調整維度以符合 Conv1D (N, C, L)
        # C (通道數) = vocab_size, L (長度) = seq_len
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(1, 0)  # 轉換為 (vocab_size, seq_len)
        label_tensor = torch.tensor(label, dtype=torch.long) # 分類任務標籤通常為 long 型別
        return sequence_tensor, label_tensor

def load_and_preprocess_data(promoter_file_list, non_promoter_file_list, data_dir=config.DATA_DIR, seq_len=config.SEQ_LEN):
    """
    載入 FASTA 檔案中的序列，進行獨熱編碼，並賦予標籤。

    Args:
        promoter_file_list (list): 包含啟動子序列 FASTA 檔名的列表。
        non_promoter_file_list (list): 包含非啟動子序列 FASTA 檔名的列表。
        data_dir (str, optional): FASTA 檔案所在的目錄。預設為 config.DATA_DIR。
        seq_len (int, optional): 預期的序列長度。預設為 config.SEQ_LEN。
                                 長度不符的序列將被跳過。

    Returns:
        tuple: (all_sequences_encoded, all_labels)
               all_sequences_encoded (numpy.ndarray): 所有獨熱編碼後的序列。
               all_labels (numpy.ndarray): 對應的標籤 (啟動子為 1，非啟動子為 0)。
    """
    all_sequences_encoded = []
    all_labels = []

    # 載入啟動子序列 (標籤為 1)
    for fname in promoter_file_list:
        fpath = os.path.join(data_dir, fname)
        sequences = read_fasta(fpath)
        for seq in sequences:
            if len(seq) == seq_len:  # 確保序列長度符合設定
                all_sequences_encoded.append(one_hot_encode(seq, seq_len))
                all_labels.append(1)
            else:
                print(f"警告: 檔案 {fname} 中序列長度 ({len(seq)}) 不符預期 ({seq_len})，已跳過。")

    # 載入非啟動子序列 (標籤為 0)
    for fname in non_promoter_file_list:
        fpath = os.path.join(data_dir, fname)
        sequences = read_fasta(fpath)
        for seq in sequences:
            if len(seq) == seq_len:  # 確保序列長度符合設定
                all_sequences_encoded.append(one_hot_encode(seq, seq_len))
                all_labels.append(0)
            else:
                print(f"警告: 檔案 {fname} 中序列長度 ({len(seq)}) 不符預期 ({seq_len})，已跳過。")

    return np.array(all_sequences_encoded), np.array(all_labels)

if __name__ == '__main__':
    # 此區塊用於測試 data_utils.py 中的函數功能
    print("測試 data_utils.py...")
    prom_files = config.PROMOTER_FILES
    non_prom_files = config.NON_PROMOTER_FILES

    X, y = load_and_preprocess_data(prom_files, non_prom_files)
    print(f"總共載入序列數量: {X.shape[0]}")
    print(f"獨熱編碼後的序列形狀: {X.shape}")  # 應為 (樣本數, 序列長度, 詞彙大小)
    print(f"標籤形狀: {y.shape}")
    print(f"啟動子數量: {np.sum(y == 1)}")
    print(f"非啟動子數量: {np.sum(y == 0)}")

    # 測試 Dataset 和 DataLoader
    if X.shape[0] > 0:
        dataset = DNASequenceDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        sample_batch_X, sample_batch_y = next(iter(dataloader))
        print(f"DataLoader 批次 X 形狀: {sample_batch_X.shape}")  # 應為 (批次大小, 詞彙大小, 序列長度)
        print(f"DataLoader 批次 y 形狀: {sample_batch_y.shape}")
    else:
        print("沒有載入任何資料，無法測試 DataLoader。請檢查 FASTA 檔案和路徑。")

    # 測試 one_hot_decode
    if X.shape[0] > 0:
        # 取前兩個獨熱編碼序列進行解碼測試
        sample_encoded = X[0:2]  # 形狀 (2, seq_len, vocab_size)
        decoded_strings = one_hot_decode(sample_encoded)
        print(f"\n測試 one_hot_decode:")
        for i, s_str in enumerate(decoded_strings):
            print(f"解碼序列 {i+1} (前60bp): {s_str[:60]}...")
        # 這裡可以進一步與原始 FASTA 檔案中的序列進行對照驗證
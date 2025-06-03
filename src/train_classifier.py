import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,           # 用於計算 AUC-ROC
    average_precision_score, # 用於計算 AUC-PR (平均精確率)
    matthews_corrcoef,       # 馬修斯相關係數 (MCC)
    cohen_kappa_score,       # 科恩 Kappa 係數
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import config
from cnn_model import CNNClassifier # 導入 CNN 分類器模型
from data_utils import load_and_preprocess_data, DNASequenceDataset # 導入資料處理工具

# --- 繪圖函數 ---
def plot_classifier_curves(history_dict, save_path_prefix_str):
    """
    繪製並儲存分類器訓練過程中的損失和準確率曲線。

    Args:
        history_dict (dict): 包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 的歷史記錄字典。
        save_path_prefix_str (str): 儲存圖檔的路徑前綴 (不含副檔名)。
    """
    # 繪製損失曲線 (訓練集 vs 驗證集)
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(history_dict['train_loss'], label="Train Loss")
    plt.plot(history_dict['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path_prefix_str}_loss.png")
    plt.close()
    print(f"分類器損失曲線已儲存到: {save_path_prefix_str}_loss.png")

    # 繪製準確率曲線 (訓練集 vs 驗證集)
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(history_dict['train_acc'], label="Train Accuracy (approx.)") # 訓練準確率可能為批次平均
    plt.plot(history_dict['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{save_path_prefix_str}_accuracy.png")
    plt.close()
    print(f"分類器準確率曲線已儲存到: {save_path_prefix_str}_accuracy.png")


def evaluate_model(model_obj, dataloader_obj, device_obj, loss_fn_obj, is_training_eval=False):
    """
    在給定的 DataLoader 上評估模型性能。

    Args:
        model_obj (nn.Module): 要評估的模型。
        dataloader_obj (DataLoader): 包含評估數據的 DataLoader。
        device_obj (torch.device): 計算設備。
        loss_fn_obj (nn.Module): 損失函數。
        is_training_eval (bool): 若為 True，則為訓練過程中的驗證，僅返回簡化指標 (用於早停)。
                               若為 False，則為最終測試，返回完整指標。

    Returns:
        如果 is_training_eval is True:
            tuple: (avg_loss, accuracy, f1_macro_from_report)
        如果 is_training_eval is False:
            tuple: (avg_loss, accuracy, report_str, cm, additional_metrics_dict)
    """
    model_obj.eval() # 設定為評估模式
    total_loss = 0
    all_preds_list = []         # 儲存所有預測標籤
    all_labels_list = []        # 儲存所有真實標籤
    all_probs_class1_list = []  # 儲存正類別 (類別1) 的預測機率，用於 AUC 計算

    with torch.no_grad(): # 評估時不需要計算梯度
        for sequences, labels in dataloader_obj:
            sequences, labels = sequences.to(device_obj), labels.to(device_obj)
            outputs_logits = model_obj(sequences) # 模型輸出 logits
            loss = loss_fn_obj(outputs_logits, labels)
            total_loss += loss.item()

            # 計算類別機率 (假設是二分類，取類別 1 的機率)
            # 如果模型輸出 logits, 需要先通過 softmax
            probs_all_classes = torch.softmax(outputs_logits, dim=1)
            all_probs_class1_list.extend(probs_all_classes[:, 1].cpu().numpy()) # 儲存正類別的機率

            _, predicted_labels = torch.max(outputs_logits.data, 1) # 從 logits 中獲取預測類別
            all_preds_list.extend(predicted_labels.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader_obj) # 平均損失
    accuracy = accuracy_score(all_labels_list, all_preds_list) # 準確率

    if is_training_eval:
        # 在訓練時的評估，僅返回損失、準確率和宏平均 F1 分數 (用於早停)
        try:
            # classification_report 計算 P, R, F1 等指標
            report_dict = classification_report(all_labels_list, all_preds_list, output_dict=True, zero_division=0)
            f1_macro_from_report = report_dict['macro avg']['f1-score']
        except Exception:
            # 如果 classification_report 出錯 (例如某類別完全沒有預測)，手動計算 F1
            _, _, f1_scores_per_class, _ = precision_recall_fscore_support(all_labels_list, all_preds_list, average=None, zero_division=0)
            f1_macro_from_report = np.mean(f1_scores_per_class) if len(f1_scores_per_class) > 0 else 0.0
        return avg_loss, accuracy, f1_macro_from_report

    # 對於最終測試，生成完整報告和額外指標
    report_str = classification_report(all_labels_list, all_preds_list, target_names=['Non-Promoter (0)', 'Promoter (1)'], digits=4, zero_division=0)
    cm = confusion_matrix(all_labels_list, all_preds_list) # 混淆矩陣

    # 計算額外指標
    # 確保混淆矩陣是 2x2 (適用於二分類)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0) # 真負、偽正、偽負、真正

    specificity_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # 特異度 (真負率)
    false_positive_rate_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # 偽正率
    false_negative_rate_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # 偽負率

    # AUC-ROC: 需要類別 1 的機率
    auc_roc = 0.0
    # 確保標籤中至少有兩個類別，且機率列表與標籤列表長度一致
    if len(np.unique(all_labels_list)) > 1 and len(all_probs_class1_list) == len(all_labels_list):
        try:
            auc_roc = roc_auc_score(all_labels_list, all_probs_class1_list)
        except ValueError as e:
            print(f"計算 AUC-ROC 時出錯: {e}。可能所有樣本都屬於同一預測類別或真實類別。")
            auc_roc = 0.0 # 或設為 np.nan
    else:
        print("警告: 計算 AUC-ROC 時，標籤中只有一個類別或機率列表不匹配。AUC-ROC 設為 0。")

    # AUC-PR (Average Precision Score): 需要類別 1 的機率
    auc_pr = 0.0
    if len(np.unique(all_labels_list)) > 1 and len(all_probs_class1_list) == len(all_labels_list):
        try:
            auc_pr = average_precision_score(all_labels_list, all_probs_class1_list)
        except ValueError as e:
            print(f"計算 AUC-PR 時出錯: {e}。可能所有樣本都屬於同一預測類別或真實類別。")
            auc_pr = 0.0 # 或設為 np.nan
    else:
        print("警告: 計算 AUC-PR 時，標籤中只有一個類別或機率列表不匹配。AUC-PR 設為 0。")

    mcc = matthews_corrcoef(all_labels_list, all_preds_list) # 馬修斯相關係數
    cohen_kappa = cohen_kappa_score(all_labels_list, all_preds_list) # 科恩 Kappa 係數

    additional_metrics_dict = {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "MCC": mcc,
        "Cohen's Kappa": cohen_kappa,
        "Specificity (TNR)": specificity_tnr,
        "False Positive Rate (FPR)": false_positive_rate_fpr,
        "False Negative Rate (FNR)": false_negative_rate_fnr,
    }

    return avg_loss, accuracy, report_str, cm, additional_metrics_dict

def train_one_scenario(scenario_name_str, X_train_data, y_train_data, X_val_data, y_val_data, X_test_data, y_test_data, model_save_path_str):
    """
    針對特定場景 (例如，僅真實資料、增強資料) 訓練並評估一個 CNN 分類器。

    Args:
        scenario_name_str (str): 當前訓練場景的名稱 (用於日誌和儲存)。
        X_train_data, y_train_data: 訓練集的特徵和標籤。
        X_val_data, y_val_data: 驗證集的特徵和標籤。
        X_test_data, y_test_data: 測試集的特徵和標籤。
        model_save_path_str (str): 最佳模型的儲存路徑。

    Returns:
        str: 包含該場景測試結果的摘要字串。
    """
    print(f"\n--- 開始訓練場景: {scenario_name_str} ---")
    print(f"訓練集大小: {X_train_data.shape[0]} (啟動子: {np.sum(y_train_data==1)}, 非啟動子: {np.sum(y_train_data==0)})")
    print(f"驗證集大小: {X_val_data.shape[0]}")
    print(f"測試集大小: {X_test_data.shape[0]}")

    device = config.DEVICE # 使用 config 中設定的設備

    # 創建 Dataset 和 DataLoader
    train_dataset = DNASequenceDataset(X_train_data, y_train_data)
    val_dataset = DNASequenceDataset(X_val_data, y_val_data)
    test_dataset = DNASequenceDataset(X_test_data, y_test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=True, drop_last=True) # drop_last 確保批次大小一致
    val_loader = DataLoader(val_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=False)

    # 初始化模型、優化器和損失函數
    model = CNNClassifier(seq_len=config.SEQ_LEN, vocab_size=config.VOCAB_SIZE, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LR)

    # 計算類別權重以處理類別不平衡問題 (如果存在)
    counts_per_class = np.bincount(y_train_data) # 獲取每個類別的樣本數
    if len(counts_per_class) < 2 or counts_per_class[0] == 0 or counts_per_class[1] == 0: # 檢查是否有類別缺失或樣本數為零
        print("警告: 訓練集中某個類別樣本數為0或只有一個類別，不使用類別加權。")
        criterion = nn.CrossEntropyLoss() # 標準交叉熵損失
    else:
        num_samples_total = len(y_train_data)
        # 計算權重: weight_class_i = total_samples / (num_classes * samples_in_class_i)
        class_weights_values = [num_samples_total / (2 * counts_per_class[0]),
                                num_samples_total / (2 * counts_per_class[1])]
        print(f"計算得到的類別權重 (針對類別 0, 類別 1): {class_weights_values}")
        class_weights_tensor = torch.tensor(class_weights_values, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # 帶權重的交叉熵損失

    # 訓練歷史記錄
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 早停 (Early Stopping) 參數
    best_val_f1_score = 0.0  # 追蹤最佳驗證集 F1 分數
    epochs_no_improve = 0    # 連續未提升的 epoch 次數
    patience = 10            # 早停的容忍度 (連續多少 epoch 未提升則停止)

    # 開始訓練迴圈
    for epoch in range(config.CNN_EPOCHS):
        model.train() # 設定為訓練模式
        running_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad() # 清除梯度
            outputs_logits = model(sequences) # 前向傳播
            loss = criterion(outputs_logits, labels) # 計算損失
            loss.backward() # 反向傳播
            optimizer.step() # 更新權重
            running_loss += loss.item()

            # 計算訓練集上的近似準確率 (每個批次的)
            _, predicted = torch.max(outputs_logits.data, 1)
            total_train_samples += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy_approx = correct_train_preds / total_train_samples

        # 在驗證集上評估模型，獲取損失、準確率和宏平均 F1 分數
        val_loss, val_accuracy, val_f1_macro = evaluate_model(model, val_loader, device, criterion, is_training_eval=True)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy_approx)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{config.CNN_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc (approx): {train_accuracy_approx:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Macro F1: {val_f1_macro:.4f}")

        # 早停邏輯：如果驗證集 F1 分數提升，則儲存模型並重置計數器
        if val_f1_macro > best_val_f1_score:
            best_val_f1_score = val_f1_macro
            # 確保模型儲存目錄存在
            if not os.path.exists(os.path.dirname(model_save_path_str)) and os.path.dirname(model_save_path_str):
                os.makedirs(os.path.dirname(model_save_path_str))
            torch.save(model.state_dict(), model_save_path_str)
            print(f"在 epoch {epoch+1} 儲存較佳模型到 {model_save_path_str} (Val Macro F1: {best_val_f1_score:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"連續 {patience} 個 epoch 驗證集 Macro F1 未提升，提早停止訓練。")
                break # 跳出訓練迴圈

    print(f"訓練完成 ({scenario_name_str})。載入最佳模型進行最終測試...")
    model.load_state_dict(torch.load(model_save_path_str)) # 載入早停時儲存的最佳模型
    # 在測試集上進行最終評估
    test_loss, test_accuracy, test_report_str, test_cm, additional_metrics = evaluate_model(model, test_loader, device, criterion)

    # 構建結果摘要字串
    result_summary_str = f"\n--- {scenario_name_str} 測試結果 ---\n"
    result_summary_str += f"測試損失: {test_loss:.4f}\n"
    result_summary_str += f"測試準確率: {test_accuracy:.4f}\n"
    result_summary_str += "分類報告:\n" + test_report_str + "\n"
    result_summary_str += "混淆矩陣:\n" + str(test_cm) + "\n"
    result_summary_str += "額外指標:\n"
    for metric_name, metric_value in additional_metrics.items():
        result_summary_str += f"  {metric_name}: {metric_value:.4f}\n"
    result_summary_str += "\n"

    print(result_summary_str) # 印出摘要

    # 繪製並儲存該場景的訓練曲線圖
    plot_save_prefix = os.path.join(config.RESULTS_DIR, scenario_name_str.replace(" ", "_").replace("(", "").replace(")", ""))
    plot_classifier_curves(history, plot_save_prefix)

    return result_summary_str


def main():
    """分類器訓練與評估主流程。"""
    # 設定隨機種子以確保可重現性
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    # 確保結果儲存目錄存在
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    print("載入所有真實資料...")
    # 載入並預處理所有真實數據 (啟動子與非啟動子)
    X_real_all, y_real_all = load_and_preprocess_data(
        promoter_file_list=config.PROMOTER_FILES,
        non_promoter_file_list=config.NON_PROMOTER_FILES,
        data_dir=config.DATA_DIR,
        seq_len=config.SEQ_LEN
    )
    print(f"總真實資料量: {X_real_all.shape[0]} (啟動子: {np.sum(y_real_all==1)}, 非啟動子: {np.sum(y_real_all==0)})")

    if X_real_all.shape[0] == 0:
        print("錯誤：沒有載入任何真實資料。請檢查您的 FASTA 檔案和路徑。分類器訓練中止。")
        return

    # 將真實數據集分割為訓練集、驗證集和測試集 (例如 80% 訓練, 10% 驗證, 10% 測試)
    # stratify=y_real_all 確保分割時各類別比例保持一致
    # 第一次分割：分出 10% 作為測試集
    X_real_temp, X_real_test, y_real_temp, y_real_test = train_test_split(
        X_real_all, y_real_all, test_size=0.10, random_state=config.RANDOM_SEED, stratify=y_real_all
    )
    # 第二次分割：從剩餘的 90% (X_real_temp) 中分出約 11.1% (即原始的 10%) 作為驗證集
    # (1/9 of 90% is 10% of total)
    X_real_train, X_real_val, y_real_train, y_real_val = train_test_split(
        X_real_temp, y_real_temp, test_size=(1/9), random_state=config.RANDOM_SEED, stratify=y_real_temp
    )
    print(f"真實訓練集大小: {X_real_train.shape[0]} (啟動子: {np.sum(y_real_train==1)}) (約佔總資料 80%)")
    print(f"真實驗證集大小: {X_real_val.shape[0]} (啟動子: {np.sum(y_real_val==1)}) (約佔總資料 10%)")
    print(f"真實測試集大小: {X_real_test.shape[0]} (啟動子: {np.sum(y_real_test==1)}) (約佔總資料 10%)")

    # --- 場景 A: 僅使用真實資料訓練分類器 ---
    report_A_results = train_one_scenario(
        scenario_name_str="真實資料 (Real Data Only)",
        X_train_data=X_real_train, y_train_data=y_real_train,
        X_val_data=X_real_val, y_val_data=y_real_val,
        X_test_data=X_real_test, y_test_data=y_real_test,
        model_save_path_str=config.CNN_MODEL_REAL_DATA_PATH
    )

    print("\n載入合成資料...")
    if not os.path.exists(config.SYNTHETIC_DATA_PATH):
        print(f"錯誤：找不到合成資料檔案 {config.SYNTHETIC_DATA_PATH}。請先執行 generate_sequences.py。")
        print("將僅儲存場景 A (真實資料) 的結果。")
        # 如果沒有合成資料，只寫入場景 A 的報告
        with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("實驗結果總結\n")
            f.write("===================================\n")
            f.write(report_A_results)
            f.write("\n===================================\n")
            f.write("場景 B (增強資料) 未執行，因為找不到合成資料。\n")
        print(f"\n場景 B 未執行。結果已儲存到 {config.CLASSIFICATION_REPORT_PATH}")
        return # 結束程式

    # 載入合成資料 (假設已由 generate_sequences.py 生成)
    synthetic_data_dict = torch.load(config.SYNTHETIC_DATA_PATH)
    X_synthetic_promoters = synthetic_data_dict['sequences'].numpy() # 獲取合成序列
    y_synthetic_promoters = synthetic_data_dict['labels'].numpy()   # 獲取合成序列標籤 (應全為1)

    print(f"已載入 {X_synthetic_promoters.shape[0]} 筆合成啟動子資料。")

    # 確保合成資料的標籤都是 1 (代表啟動子)
    if not np.all(y_synthetic_promoters == 1):
        print(f"警告: 合成資料中包含非啟動子標籤 (非1)。預期所有合成資料都應為啟動子。將繼續，但請檢查生成過程。")
        # y_synthetic_promoters = np.ones_like(y_synthetic_promoters) # 或者可以選擇強制將標籤設為1

    # --- 場景 B: 使用增強資料 (真實啟動子 + 合成啟動子 + 真實非啟動子) 訓練分類器 ---
    # 將真實訓練集中的啟動子與合成啟動子合併，非啟動子保持不變
    # 注意：這裡的 y_real_train 包含 0 和 1，X_synthetic_promoters 的標籤 y_synthetic_promoters 應全為 1
    X_augmented_train = np.concatenate((X_real_train, X_synthetic_promoters), axis=0)
    y_augmented_train = np.concatenate((y_real_train, y_synthetic_promoters), axis=0)

    # 打亂增強後的訓練集順序
    shuffle_indices = np.random.permutation(len(X_augmented_train))
    X_augmented_train = X_augmented_train[shuffle_indices]
    y_augmented_train = y_augmented_train[shuffle_indices]

    # 驗證集和測試集保持不變，只使用真實資料，以公平比較模型性能
    report_B_results = train_one_scenario(
        scenario_name_str="增強資料 (Real + Synthetic Promoters)",
        X_train_data=X_augmented_train, y_train_data=y_augmented_train,
        X_val_data=X_real_val, y_val_data=y_real_val, # 驗證集使用原始真實數據
        X_test_data=X_real_test, y_test_data=y_real_test, # 測試集使用原始真實數據
        model_save_path_str=config.CNN_MODEL_AUGMENTED_DATA_PATH
    )

    # 將所有場景的結果寫入同一個報告檔案
    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("實驗結果總結\n")
        f.write("===================================\n")
        f.write(report_A_results)
        f.write("\n===================================\n")
        f.write(report_B_results)
        f.write("\n===================================\n")
    print(f"\n所有訓練和評估完成。結果已儲存到 {config.CLASSIFICATION_REPORT_PATH}")

if __name__ == "__main__":
    main()
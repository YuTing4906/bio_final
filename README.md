# 基於 GAN 之啟動子序列擴增及其在 CNN 分類應用上之效能評估

本專案利用 WGAN-GP 產生擬真的啟動子序列（Promoters），並透過 CNN 分類器，比較「真實資料 vs. 真實+生成資料」這兩種情況下的分類效果。我們建立了一套流程，從資料生成、品質驗證，到最終分類任務，**目的是提升分類器在啟動子辨識上的表現**。


## 📁 專案結構

```
bio_final/
├── data/                                # 存放 FASTA 序列檔
│   ├── Mouse_non_nonprom_big.fa
│   ├── Mouse_nonprom.fa
│   ├── Mouse_tata.fa
│   ├── Mouse_tata_dbtss.fa
│   └── synthetic_promoters.pt           # 儲存生成的合成資料
│
├── src/                                 # 程式碼檔案
│   ├── config.py                        # 設定檔（路徑、超參數等）
│   ├── data_utils.py                    # 資料讀取、預處理、one-hot encoding
│   ├── gan_models.py                    # GAN 生成器 (Generator) 與判別器 (Discriminator) PyTorch 模型
│   ├── train_gan.py                     # 訓練 GAN 的腳本
│   ├── generate_sequences.py            # 使用訓練好的 GAN 生成合成序列的腳本
│   ├── cnn_model.py                     # CNN 分類器 PyTorch 模型
│   ├── train_classifier.py              # 訓練與評估分類器的腳本
│   ├── validate_gan.py                  # GAN 生成資料品質驗證腳本，包含 k-mer 分析、GC 含量分析等
│   └── main.py                          # 主程式，協調整個工作流程的執行
│
├── models/                              # 儲存訓練好的模型
│   ├── gan_generator.pth
│   ├── gan_critic.pth
│   ├── cnn_real_data.pth
│   └── cnn_augmented_data.pth
│
├── results/                             # 儲存評估指標、圖表等
│   ├── classification_report.txt        # CNN 分類器的詳細效能評估報告
│   ├── gan_validation_summary.txt       # GAN 驗證結果總結
│   ├── wgan_gp_loss_curves.png          # GAN 訓練損失曲線圖
│   └── ...
│
└── README.md                            # 專案說明文件
```

## 🚀 快速開始
### 1. 安裝環境
```
pip install torch torchvision
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install umap-learn
```

本專案相容於 Python 3.11 以上版本。若系統配備 GPU（如 NVIDIA CUDA），可加快訓練與資料生成的速度。

### 2.放置資料
將 FASTA 格式的 DNA 序列檔案放入 `../data/` 目錄下，檔名需與 config.py 中的設定一致：

```python=
PROMOTER_FILES = ["Mouse_non_tata.fa", "Mouse_tata.fa", "Mouse_tata_dbtss.fa"]
NON_PROMOTER_FILES = ["Mouse_non_nonprom_big.fa", "Mouse_nonprom.fa"]
```

本專案使用的資料集來源 : https://github.com/solovictor/CNNPromoterData


### 3. 執行完整流程
```bash=
cd src/
python main.py
```

流程包含：
* 訓練 WGAN-GP (`train_gan.py`)
* 生成合成資料 (`generate_sequences.py`)
* 驗證生成資料品質（JS Divergence、GC含量、熵等）(`validate_gan.py`)
* CNN 分類器訓練與測試（比較真實 vs. 增強資料）(`train_classifier.py`)


## 🧬 生成資料驗證指標
* k-mer 分佈相似度 (Jensen Shannon Divergence, JSD)
* GC 含量分析 (GC Content Analysis)
* 核苷酸偏好性熱圖 (Nucleotide Preference Heatmap)
* 每個位置的熵 (Per-Position Entropy)
* 降維視覺化分析 (PCA, t-SNE, UMAP)
* 可辨識性測試 (模型是否能分辨真實與合成資料)

所有相關圖表皆儲存於 `../results/gan_validation/` 資料夾。


## 📋 分類器效能評估指標
* Accuracy
* Precision / Recall / F1-score
* AUC-ROC / AUC-PR
* Cohen’s Kappa / MCC
* Confusion Matrix
* FPR / FNR / TNR

所有分類結果與圖表皆儲存於 `../results/`。


## 🧠 模型架構摘要
* WGAN-GP Generator
    * 基於 1D 反卷積的神經網路 (CNN) 架構
    * 輸入為潛在向量，輸出為每個核苷酸位置的機率分佈 (使用 softmax 激活函數)
* CNN Classifier
    * 包含兩層 1D 卷積層 (Conv1D)、池化層 (Pooling) 和 Dropout 層
    * 最終全連接層輸出二元分類的 logits (未經 softmax 處理的數值)

## ⚠️ 重要注意事項
* 請確認 config.py 設定檔中的資料路徑是否正確，並且所需的資料檔案是否已存在。
* 模型訓練可能需要較長時間，尤其 GAN 訓練和 t-SNE 視覺化。建議使用 GPU 以加速訓練。
* config.py 設定檔中包含許多可調整的參數，以優化模型效能。
* validate_gan.py 腳本提供對生成序列的全面評估，包括 k-mer 分析、GC 含量分析等，對於評估 GAN 的生成能力至關重要。


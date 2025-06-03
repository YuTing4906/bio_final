import os
import sys


try:
    import config # 嘗試導入 config.py 以測試路徑是否正確
except ModuleNotFoundError:
    # 如果導入失敗，可能是因為執行腳本時的工作目錄不正確。
    print("錯誤：無法導入 src.config。請確認您在 'src' 目錄下執行此腳本，")
    print("或者您的 PYTHONPATH 環境變數已正確設定。")
    print("建議執行方式: 首先 `cd` 到 `src` 目錄，然後執行 `python main.py`")
    sys.exit(1) # 導入失敗則退出程式
import train_gan         # GAN 訓練模組
import generate_sequences # 合成序列生成模組
import validate_gan      # GAN 品質驗證模組
import train_classifier  # 分類器訓練與評估模組

def create_directories():
    """檢查並創建流程中所有必要的輸出目錄 (如果它們不存在的話)。"""
    # config.py 中的路徑是相對於 src 目錄的 (例如 "../data/")
    # 如果 main.py 在 src 目錄中執行，這些相對路徑是正確的
    print("檢查並創建必要的輸出目錄...")
    paths_to_create = [
        config.DATA_DIR,    # 資料目錄 (雖然通常已包含原始數據，但檢查無妨)
        config.MODELS_DIR,  # 模型儲存目錄
        config.RESULTS_DIR, # 結果與報告儲存目錄
        config.GAN_VALIDATION_PLOT_DIR # GAN 驗證圖表儲存目錄
    ]
    for dir_path in paths_to_create:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True) # exist_ok=True 避免並行創建時的競爭條件
                print(f"  已建立目錄: {dir_path}")
            except OSError as e:
                print(f"  建立目錄 {dir_path} 失敗: {e}")
                # 如果是關鍵目錄 (如 DATA_DIR) 創建失敗，可能需要中止程式
                if dir_path == config.DATA_DIR and not os.path.exists(dir_path):
                    print(f"錯誤：關鍵目錄 {config.DATA_DIR} 無法建立或不存在，程式中止。")
                    sys.exit(1)
        else:
            print(f"  目錄已存在: {dir_path}")


def main_orchestrator():
    """
    協調執行整個 DNA 序列增強與分類流程的各個階段。
    """
    print("="*50)
    print("開始執行 DNA 序列增強與分類完整流程")
    print("="*50)

    # 步驟 0: 創建所有必要的輸出目錄
    create_directories()

    # 步驟 1: 訓練 GAN 模型
    print("\n[階段 1/4] 開始訓練 GAN...")
    print("-"*30)
    try:
        train_gan.main() # 執行 GAN 訓練
        print("-"*30)
        print("[階段 1/4] GAN 訓練成功完成。")
    except Exception as e:
        print("-"*30)
        print(f"[階段 1/4] GAN 訓練過程中發生錯誤: {e}")
        print("流程中止。")
        return # GAN 訓練失敗，中止後續流程

    # 步驟 2: 使用訓練好的 GAN 生成合成序列
    print("\n[階段 2/4] 開始生成合成序列...")
    print("-"*30)
    try:
        generate_sequences.main() # 執行合成序列生成
        print("-"*30)
        print("[階段 2/4] 合成序列生成成功完成。")
    except Exception as e:
        print("-"*30)
        print(f"[階段 2/4] 合成序列生成過程中發生錯誤: {e}")
        print("流程中止。")
        return # 合成序列生成失敗，中止後續流程

    # 步驟 3: 驗證 GAN 生成的資料品質
    print("\n[階段 3/4] 開始驗證 GAN 生成資料品質...")
    print("-"*30)
    try:
        validate_gan.main() # 執行 GAN 品質驗證
        print("-"*30)
        print("[階段 3/4] GAN 品質驗證成功完成。")
    except Exception as e:
        print("-"*30)
        print(f"[階段 3/4] GAN 品質驗證過程中發生錯誤: {e}")
        # GAN 品質驗證失敗不一定需要中止整個流程，可以選擇繼續
        print("GAN 品質驗證出現問題，但流程將繼續執行分類器訓練...")


    # 步驟 4: 訓練與評估分類器 (使用真實資料及增強資料)
    print("\n[階段 4/4] 開始訓練與評估分類器...")
    print("-"*30)
    try:
        train_classifier.main() # 執行分類器訓練與評估
        print("-"*30)
        print("[階段 4/4] 分類器訓練與評估成功完成。")
    except Exception as e:
        print("-"*30)
        print(f"[階段 4/4] 分類器訓練與評估過程中發生錯誤: {e}")
        print("流程中止。")
        return # 分類器訓練失敗，流程結束

    # 流程全部執行完畢總結
    print("\n" + "="*50)
    print("完整流程已全部執行完畢。")
    print(f"  模型已儲存於: {os.path.abspath(config.MODELS_DIR)}")
    print(f"  結果報告已儲存於: {os.path.abspath(config.RESULTS_DIR)}")
    print(f"  GAN 驗證圖表已儲存於: {os.path.abspath(config.GAN_VALIDATION_PLOT_DIR)}")
    if os.path.exists(config.SYNTHETIC_DATA_PATH): # 檢查合成資料是否存在
        print(f"  合成資料已儲存於: {os.path.abspath(config.SYNTHETIC_DATA_PATH)}")
    print("="*50)

if __name__ == "__main__":
    main_orchestrator()
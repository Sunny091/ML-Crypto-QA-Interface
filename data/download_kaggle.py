"""
Kaggle 資料集下載腳本

Dataset: Top 10 Cryptocurrencies Historical Dataset
https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset

使用方式：
1. 安裝 kagglehub: pip install kagglehub
2. 執行: python -m data.download_kaggle
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 幣種對應的檔案名稱（Kaggle dataset 格式）
COIN_FILES = {
    "BTC": "bitcoin.csv",
    "ETH": "ethereum.csv",
    "SOL": "solana.csv",
    "XRP": "XRP.csv",
    "BNB": "BNB.csv",
    "ADA": "cardano.csv",
    "DOGE": "dogecoin.csv",
}


def download_from_kaggle() -> str:
    """從 Kaggle 下載資料集"""
    try:
        import kagglehub

        print("正在從 Kaggle 下載資料集...")
        path = kagglehub.dataset_download(
            "kaushiksuresh147/top-10-cryptocurrencies-historical-dataset"
        )
        print(f"下載完成！資料存放於: {path}")
        return path

    except ImportError:
        print("請先安裝 kagglehub: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        print(f"下載失敗: {e}")
        print("\n請手動下載:")
        print("1. 前往 https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset")
        print("2. 下載資料集")
        sys.exit(1)


def copy_to_project(kaggle_path: str):
    """複製資料到專案目錄"""
    kaggle_path = Path(kaggle_path)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 搜尋所有 CSV 檔案（包含子目錄）
    csv_files = list(kaggle_path.rglob("*.csv"))
    print(f"找到 {len(csv_files)} 個 CSV 檔案")

    for src_file in csv_files:
        dst_file = DATA_RAW_DIR / src_file.name
        if not dst_file.exists():
            shutil.copy(src_file, dst_file)
            print(f"  複製: {src_file.name}")
        else:
            print(f"  跳過（已存在）: {src_file.name}")


def check_data_exists() -> bool:
    """檢查資料是否已存在"""
    if not DATA_RAW_DIR.exists():
        return False

    # 檢查是否有主要的幣種資料
    for coin, filename in COIN_FILES.items():
        if (DATA_RAW_DIR / filename).exists():
            return True
    return False


def list_available_coins():
    """列出可用的幣種"""
    if not DATA_RAW_DIR.exists():
        print("資料目錄不存在")
        return

    csv_files = list(DATA_RAW_DIR.glob("*.csv"))
    print(f"\n可用的資料檔案 ({len(csv_files)} 個):")
    for f in sorted(csv_files):
        print(f"  - {f.name}")


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if check_data_exists():
        print("資料集已存在")
        list_available_coins()
        return

    # 下載資料
    kaggle_path = download_from_kaggle()

    # 複製到專案目錄
    copy_to_project(kaggle_path)

    list_available_coins()
    print("\n下載完成！")


if __name__ == "__main__":
    main()

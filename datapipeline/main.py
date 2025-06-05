# main.py
import sys
import os
from src.data_loaders.temp_loader import main as load_temp_main
from src.data_loaders.pm10_loader import main as load_pm10_main
from src.data_processors.eda_processor import main as eda_main
from src.cloud.s3_uploader import upload_to_s3

def main():
    try:
        print("Loading temperature data...")
        load_temp_main()

        print("Loading PM10 data...")
        load_pm10_main()

        print("Running EDA and saving results...")
        temp_dates, pm10_dates = eda_main()  # 날짜 리스트 반환

        print("Uploading files to S3...")
        upload_to_s3(temp_dates, pm10_dates)  # 날짜별 업로드

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    print("Data processing and S3 upload completed successfully!")

if __name__ == "__main__":
    main()

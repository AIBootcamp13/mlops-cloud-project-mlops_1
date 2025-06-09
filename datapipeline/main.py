# datapipeline/main.py

import sys
from src.cloud.s3_uploader import upload_to_s3_for_recent_days

def main():
    try:
        print("Running EDA and uploading results...")
        upload_to_s3_for_recent_days()  # 이 한 줄이면 충분

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    print("Data processing and S3 upload completed successfully!")

if __name__ == "__main__":
    main()

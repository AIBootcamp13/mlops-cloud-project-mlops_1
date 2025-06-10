from src.cloud.s3_uploader import upload_to_s3_for_recent_days

def main():
    upload_to_s3_for_recent_days()  # 단순 실행만

if __name__ == '__main__':
    main()
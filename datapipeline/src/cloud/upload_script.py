import argparse
from src.cloud.s3_uploader import upload_to_s3

def parse_args():
    parser = argparse.ArgumentParser(description="Upload processed weather data to S3.")
    parser.add_argument('--temp_dates', required=True, help='Comma-separated list of temperature dates (e.g. 2025-06-01,2025-06-02)')
    parser.add_argument('--pm10_dates', required=True, help='Comma-separated list of PM10 dates (e.g. 2025-06-01,2025-06-02)')
    return parser.parse_args()

def main():
    args = parse_args()
    temp_dates = args.temp_dates.split(',')
    pm10_dates = args.pm10_dates.split(',')
    upload_to_s3(temp_dates, pm10_dates)

if __name__ == '__main__':
    main()
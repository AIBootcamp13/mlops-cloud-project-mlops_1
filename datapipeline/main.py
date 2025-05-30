import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts import loadTemp, loadAD, eda

def main():
    # 기온 데이터 다운로드 및 전처리
    print("Loading temperature data...")
    loadTemp.main()

    # 미세먼지 데이터 다운로드 및 전처리
    print("Loading PM10 data...")
    loadAD.main()

    # EDA 및 모델링 데이터 저장
    print("Running EDA and saving results...")
    eda.main()

    print("Datapipeline completed successfully!")

if __name__ == "__main__":
    main()
# MLOps Project

<br>

# 프로젝트 소개
## <서울시의 일일 기온 및 미세먼지를 예측하여 웹 페이지에 자동 업데이트하는 AI 기반 MLOps 프로젝트>
● 목표 : Datapipeline, Modeling, Serving 컴포넌트의 자동화 구현 (데이터 전처리부터 모델 서빙까지의 경험)

● 프로젝트 진행 기간: 2025. 05. 26 - 2025. 06. 10

● 작업:    
    - 데이터 파이프라인 자동화    
    - 모델 학습 및 평가 자동화    
    - 배치 서빙 자동화    
    
● 상세 기능:    
    - 매일 오전 4시, 기상청 API로부터 기온과 미세먼지 데이터를 수집      
    - 수집된 데이터를 전처리 및 EDA 후 S3 버킷에 저장      
    - 저장된 데이터를 바탕으로 모델 예측 수행      
    - FastAPI + React를 통해 사용자에게 예측 결과를 시각화 제공    

<br>

# 팀 구성원

| ![박진일](https://github.com/user-attachments/assets/551cb3b6-c019-4e50-a35b-1843d3e474a3) | ![이민우](https://github.com/user-attachments/assets/b71c7815-e0c7-407f-b593-121e11c61d05) | ![조은별](https://github.com/user-attachments/assets/781faa65-c309-49ee-b58a-9cc08b13b69c) | ![조재형](https://github.com/user-attachments/assets/c7d4d78c-36ae-4fb1-bbd4-105ee9d722cc) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박진일](https://github.com/UpstageAILab)             |            [이민우](https://github.com/UpstageAILab)             |            [조은별](https://github.com/UpstageAILab)             |            [조재형](https://github.com/UpstageAILab)             |
|                            팀장, PM               |                            모델링 파이프라인 자동화         |                            데이터 파이프라인 자동화                  |                            모델 서빙 자동화 및 배포                           |
|               프로젝트 기획 및 일정 관리, 팀원 간 역학 조율 및 산출물 검토                |          모델링 로직 구현, MLflow 기반 모델 관리 및 실험 관리,  Airflow 기반 성능 추적 자동화           |     기상청API 연동 -> EDA -> S3 저장 및 Airflow DAG 자동화        |    FastAPI로 배치 서빙 API 구축, React 기반 시각화 UI구현, Airflow 기반 예측 결과 저장 자동화                |


<br>

# 개발 환경 및 기술 스택    
● 주 언어 : python, FastAPI, React    

● 데이터: 기상청 API    

● 버전 및 이슈관리 : github    

● 협업 툴 : github, notion, discord    

● 자동화: Airflow, AWS S3, Docker    

● 모델링: scikit-learn, XGBoost    

<br>

# 아키텍쳐 설계
![기술스텍 아키텍처](https://i.imgur.com/il6vU8j.jpeg)

<br>

# 프로젝트 구조
```
├── datapipeline
│ ├── airflow/
│ │ ├── dags/
│ │ │ ├── init_full_eda_dag.py # 전체 EDA 실행용 DAG (최초 수동 실행용)
│ │ │ └── weather_pipeline_dag.py # 매일 자동 실행되는 주력 DAG
│ │ └── utils/
│ │ └── slack_notifier.py # Slack 알림 전송 모듈
│ ├── src/
│ │ ├── cloud/
│ │ │ ├── s3_uploader.py # S3 업로드 모듈
│ │ │ └── upload_script.py # 업로드 실행 스크립트
│ │ ├── config/
│ │ │ └── settings.py # 공통 환경 변수 설정
│ │ ├── data_loaders/
│ │ │ ├── pm10_loader.py # 미세먼지(PM10) 데이터 수집 모듈
│ │ │ └── temp_loader.py # 기온(Temperature) 데이터 수집 모듈
│ │ └── data_processors/
│ │ └── eda_processor.py # EDA 및 전처리 수행 로직
│ ├── Dockerfile # 도커 환경 정의 파일
│ ├── upload_requirements.txt # 의존성 정의 파일
│ └── main.py # 전체 로직 실행용 메인 스크립트
│
├── modeling
│ └── src/
│ ├── airflow/
│ │ ├── tasks/
│ │ │ ├── inference.py # 추론 태스크
│ │ │ ├── is_model_drift.py # 모델 드리프트 감지 태스크
│ │ │ └── train.py # 학습 태스크
│ │ ├── inference_dag.py # 추론용 DAG
│ │ ├── lstm_airflow_dag.py # LSTM 예측용 DAG
│ │ └── train_dag.py # 학습용 DAG
│ ├── ct/
│ │ └── airflow/
│ │ ├── dags/
│ │ │ ├── tasks/
│ │ │ │ ├── inference.py
│ │ │ │ ├── is_model_drift.py
│ │ │ │ └── train.py
│ │ │ ├── inference_dag.py
│ │ │ ├── train_dag.py
│ │ │ └── tutorial_dag.py
│ │ └── docker-compose.yaml # Airflow 테스트용 도커 설정
│ ├── dags/
│ │ ├── hello_airflow_dag.py # 테스트용 DAG
│ │ ├── lstm_airflow_dag.py
│ │ └── ml_development.py # ML 개발용 DAG
│ ├── inference/
│ │ └── inference.py # 추론 스크립트
│ ├── mlflow/
│ │ ├── study/ # 실험 로그 폴더
│ │ └── main.py # MLflow 실행 파일
│ ├── model/
│ │ └── lstm.py # LSTM 모델 정의
│ ├── postprocess/
│ │ └── postprocess.py # 예측 결과 후처리
│ ├── train/
│ │ └── train.py # 학습 스크립트
│ ├── trainer/
│ │ ├── airflowTrainer.py # Airflow 전용 학습기
│ │ ├── baselineTrainer.py # 베이스라인 학습기
│ │ └── trainer.py # 공통 학습기
│ └── utils/
│   ├── constant.py # 상수 모듈
│   └── utils.py # 공통 유틸 함수
│
├── serving
│    ├── data
│    │   ├── eval
│    │   └── train
│
├── docker-compose.yml
├──
└── requirement.txt
```

<br>

# 사용 데이터셋 개요
## 기상청 API 허브 (https://apihub.kma.go.kr/)

### 온도 데이터    
● 이름: 종관기상관측(ASOS)    
● 항목: 일별 평균기온, 최고기온, 최저기온    
● 기간: 1907.10.01 ~ 현재    

### 미세먼지(PM10) 데이터
● 이름: 황사관측(PM10)    
● 항목: 부유분진농도 (PM10)    
● 기간: 2008.04.28 ~ 현재    
● 생산주기: 5분 단위 수집 → 일 단위로 평균/최대/최소 집계 하여생성    

### 활용방식
● 각 날짜별로 서울 지역(108지점) 기온 및 미세먼지 정보를 수집    
● API 응답 데이터를 바탕으로 CSV 가공 후 S3 업로드 자동화    

<br>

# 구현 기능
## Datapipeline
### 데이터 수집
● request 기반 API 호출 스크립트 작성 (Python)    
● 온도/미세먼지 각각 수집 모듈 구현 → csv 저장    

### 데이터 전처리
● 기온 데이터의 -99.0 → 결측치 처리 및 시간 보간     
● 1953–1957 단절 구간 자동 누락 처리    
● 미세먼지 PM10 평균 > 90.8 → 이상치 필터링    
● PM10 최대 > 160.5 → clip 처리    

### 클라우드 연동
● AWS S3 저장: 파티셔닝 → 날짜 기반 저장    
● 모델은 S3 데이터를 기반으로 예측    
● S3 업로드 위치    
    – AWS S3 버킷: mlops-pipeline-jeb    
    
● S3 저장경로    
```
result/temperature/date=YYYY-MM/YYYY–MM-DD.csv

result/pm10/data=YYYY-MM/YYYY-MM-DD.csv
```

### 데이터파이프라인 자동화 (Airflow)
● DAG ID: weather_pipeline     
● 스케줄    
    - 매일 오전 4시 (KST 기준)    
![DAG](https://i.imgur.com/HxRSUr8.jpeg)
    - 처리 결과 -> slack 알림    
![데이터파이프라인 슬랙 알림](https://i.imgur.com/wfESasB.jpeg)    
<br>
● Task 흐름    
    1. load_temperature_data    
    2. load_pm10_data    
    3. run_eda_and_upload    
![데이터파이프라인 테스크](https://i.imgur.com/Wmq1Ksr.jpeg)

<br>

## Modeling
### 환경 구성
● AWS EC2 클라우드 환경에서 mlops 모델링 개발    
● Ai stages gpu server에서 모델 학습     
● Docker: train, inference 환경을 통일하기 위해 사용    
● Docker-compose : airflow, mlflow 등 여러 container 기반 서비스 통합 관리    

### 모델 학습 및 배포    
● lstm 기반 시계열 예측 모델 구현    
● FastAPI를 이용해 inference api 배포    

### Airflow 자동화
● 모델 학습 -> 이상치 감지 -> 트리거 기반 재학습 자동화 구축    
● DAG를 통해 일관된 재학습 루틴 구축    

 ### MLflow
● 실험별 성능 시각화 (batch size, model 종류에 따른 val loss)    
● 학습된 모델을 model registry에 등록    
● alias을 활용하여 모델 버전 관리    

### 모니터링
● slack과 연동하여 시간 모델 성능 모니터    

mlflow 를 이용한 모델 관리 및 fastapi 를 이용해 api 서빙    
![mlflow1](https://i.imgur.com/71vggwp.jpeg)  
![mlflow2](https://i.imgur.com/2OeomSD.jpeg)  
<br>

airflow 기반 모델 재학습 자동화 파이프라인    
![airflow](https://i.imgur.com/Sf4UxCx.jpeg)  
<br>

Slack과 연동하여 실시간 모델 성능 모니터링  
![slack1](https://i.imgur.com/kaFCMuH.jpeg)  
![slack2](https://i.imgur.com/u2ZhV60.jpeg)

<br>

## API & Web Serving
### 사용한 모델:
● FestAPI를 사용한 예측 API 제공      
● React를 활용한 사용자 페이지 구현    
● API 호출 예시 (기온 및 미세먼지 예측값 표시)    

### 배포 과정 :
● 사전 학습된 모델을 fastAPI 서버에서로드하여 예측 요청을 받을 수 있는 API /result를 구성합니다.    
● 사용자 웹페이지(React)에서 이 API를 호출하여 매일 새벽에 자동으로저장된 예측 결과를 시각화합니다    

### 배치 서빙을 위한 Airflow 사용:
● 모델 예측 자동화 : 매일 새벽 5시(KST 기준) Airflow DAG를 통해 S3에서 모델 다운로드 -> 예측 수행 -> 결과 저장 자동화    
● 입력 데이터 예측 및 저장 : 버킷에 저장된 최신 입력 데이터를 기반으로 기온 및 미세먼지 예측 결과를 .csv로 저장 -> 이후 FastAPI가 해당 결과를 읽어 사용자에게 제공    
<br>
![https://i.imgur.com/EXmhb8N.jpeg](https://i.imgur.com/EXmhb8N.jpeg)

<br>

## 모니터링
### 성능 개선 변화 추이
● Airflow DAG 실행 성공 여부, 로그 및 결과 파일 저장 상태 추적    
● FastAPI /result API 응답 상태 실시간 로그 모니터링    
● 예측 실패 시 빠르게 원인 파악(모델 미로드. 데이터 누락, 등)    
● 프론트에서 실시간 예측 결과 출력 및 오류 메세지 대응    

![https://i.imgur.com/vd0TciN.jpeg](https://i.imgur.com/vd0TciN.jpeg)

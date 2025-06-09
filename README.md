# MLOps Project

<br>

## 프로젝트 소개
### <서울시의 일일 기온 및 미세먼지를 예측하여 웹 페이지에 자동 업데이트하는 AI 기반 MLOps 프로젝트>
-  목표 : Datapipeline, Modeling, Serving 컴포넌트의 자동화 구현 (데이터 전처리부터 모델 서빙까지의 경험)
-  프로젝트 진행 기간: 2025. 05. 26 - 2025. 06. 10
-  주요 작업:
    - 데이터 파이프라인 자동화
    - 모델 학습 및 평가 자동화
    - 배치 서빙 자동화
- 상세 기능:
    - 매일 오전 4시, 기상청 API로부터 기온과 미세먼지 데이터를 수집  
    - 수집된 데이터를 전처리 및 EDA 후 S3 버킷에 저장  
    - 저장된 데이터를 바탕으로 모델 예측 수행  
    - FastAPI + React를 통해 사용자에게 예측 결과를 시각화 제공

<br>

## 팀 구성원

| ![박진일](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이민우](https://avatars.githubusercontent.com/u/156163982?v=4) | ![조은별](https://avatars.githubusercontent.com/u/156163982?v=4) | ![조재형](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박진일](https://github.com/UpstageAILab)             |            [이민우](https://github.com/UpstageAILab)             |            [조은별](https://github.com/UpstageAILab)             |            [조재형](https://github.com/UpstageAILab)             |
|                            팀장, PM               |                            모델링 파이프라인 자동화         |                            데이터 파이프라인 자동화                  |                            모델 서빙 자동화 및 배포                           |
|               프로젝트 기획 및 일정 관리, 팀원 간 역학 조율 및 산출물 검토                |          모델링 로직 구현, MLflow 기반 모델 관리 및 실험 관리,  Airflow 기반 성능 추적 자동화           |     기상청API 연동 -> EDA -> S3 저장 및 Airflow DAG 자동화        |    FastAPI로 배치 서빙 API 구축, React 기반 시각화 UI구현, Airflow 기반 예측 결과 저장 자동화                |


<br>

## 개발 환경 및 기술 스택
- 주 언어 : python, FastAPI, React
- 데이터: 기상청 API
- 버전 및 이슈관리 : github
- 협업 툴 : github, notion, discord
- 자동화: Airflow, AWS S3, Docker
- 모델링: scikit-learn, XGBoost

<br>

## 아키텍쳐 설계
![기술스텍_아키텍처.jpg](attachment:ef02f979-cae7-40db-8c63-459f48ad309a:기술스텍_아키텍처.jpg)

<br>

## 프로젝트 구조
```
├── datapipeline
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── modeling
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── serving
    └── data
        ├── eval
        └── train
```

<br>

## 사용 데이터셋 개요
### 기상청 API 허브 (https://apihub.kma.go.kr/)

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


## 구현 기능
### 데이터셋 수집
- request 기반 API 호출 스크립트 작성 (Python)
- 온도/미세먼지 각각 수집 모듈 구현 → csv 저장

### 데이터 전처리
- 기온 데이터의 -99.0 → 결측치 처리 및 시간 보간
- 1953–1957 단절 구간 자동 누락 처리
- 미세먼지 PM10 평균 > 90.8 → 이상치 필터링
- PM10 최대 > 160.5 → clip 처리

### 클라우드 연동
- AWS S3 저장: 파티셔닝 → 날짜 기반 저장
- 모델은 S3 데이터를 기반으로 예측
- S3 업로드 위치
    – AWS S3 버킷: mlops-pipeline-jeb
- 저장경로
```
result/temperature/date=Y
YYY-MM/YYYY–MM-DD.csv

result/pm10/data=YYYY-M
M/YYYY-MM-DD.csv
```

### 데이터파이프라인 자동화 (Airflow)
- DAG ID: weather_pipeline
- 스케줄
    - 매일 오전 4시 (KST 기준)
    - 처리 결과 -> slack 알림
![화면 캡처 2025-06-09 230545.jpg](attachment:cc7aa7be-dbaf-4105-9fa0-d6c26daedda8:화면_캡처_2025-06-09_230545.jpg)
![화면 캡처 2025-06-09 232214.jpg](attachment:117adc75-016d-4b87-9d45-9a1c34a0f803:화면_캡처_2025-06-09_232214.jpg)
- Task 흐름
    1. load_temperature_data
    2. load_pm10_data
    3. run_eda_and_upload
![화면 캡처 2025-06-09 230244.jpg](attachment:18f6b08d-3c21-42a3-b034-3a281a455689:화면_캡처_2025-06-09_230244.jpg)


### Modeling
- 환경 구성
    - AWS EC2 클라우드 환경에서 mlops 모델링 개발
    - Ai stages gpu server에서 모델 학습
    - Docker: train, inference 환경을 통일하기 위해 사용
    - Docker-compose : airflow, mlflow 등 여러 container 기반 서비스 통합 관리
- 모델 학습 및 배포
    ● lstm 기반 시계열 예측 모델 구현
    ● FastAPI를 이용해 inference api 배포
- Airflow 자동화
    ● 모델 학습 -> 이상치 감지 -> 트리거 기반 재학습 자동화 구축
    ● DAG를 통해 일관된 재학습 루틴 구축
- MLflow
    ● 실험별 성능 시각화 (batch size, model 종류에 따른 val loss)
    ● 학습된 모델을 model registry에 등록
    ● alias을 활용하여 모델 버전 관리
- 모니터링
    ● slack과 연동하여 시간 모델 성능 모니터

* mlflow 를 이용한 모델 관리 및 fastapi 를 이용해 api 서빙
![화면 캡처 2025-06-09 232414.jpg](attachment:01ad19e6-3a11-4a70-a7d8-521470818fdc:화면_캡처_2025-06-09_232414.jpg)
![화면 캡처 2025-06-09 231802.jpg](attachment:146c6161-26b2-4734-94cd-d0070813487f:화면_캡처_2025-06-09_231802.jpg)

airflow 기반 모델 재학습 자동화
파이프라인
![화면 캡처 2025-06-09 231816.jpg](attachment:37ad2f5a-7b21-4aab-8bdc-6636958796ab:화면_캡처_2025-06-09_231816.jpg)

Slack과 연동하여 실시간 모델 성능 모니터링
![화면 캡처 2025-06-09 231833.jpg](attachment:05f92169-5135-4a96-8319-3389ac0a9b3e:화면_캡처_2025-06-09_231833.jpg)
![화면 캡처 2025-06-09 231919.jpg](attachment:9802dcbf-08bd-4ca1-b961-5bbb85404303:화면_캡처_2025-06-09_231919.jpg)

### API & Web Serving
- 사용한 모델:
    - FestAPI를 사용한 예측 API 제공
    - React를 활용한 사용자 페이지 구현
    - API 호출 예시 (기온 및 미세먼지 예측값 표시)
- 배포 과정 :
    - 사전 학습된 모델을 fastAPI 서버에서로드하여 예측 요청을 받을 수 있는 API /result를 구성합니다.
    - 사용자 웹페이지(React)에서 이 API를 호출하여 매일 새벽에 자동으로저장된 예측 결과를 시각화합니다

- 배치 서빙을 위한 Airflow 사용:
    - 모델 예측 자동화 : 매일 새벽 5시(KST 기준) Airflow DAG를 통해 S3에서 모델 다운로드 -> 예측 수행 -> 결과 저장 자동화
    - 입력 데이터 예측 및 저장 : 버킷에 저장된 최신 입력 데이터를 기반으로 기온 및 미세먼지 예측 결과를 .csv로 저장합니다.
                               이후 FastAPI가 해당 결과를 읽어 사용자에게 제공합니다

![화면 캡처 2025-06-09 231513.jpg](attachment:e2f71b02-2374-4f96-ab90-0e509717b544:화면_캡처_2025-06-09_231513.jpg)

<br>

# 베이스 이미지 선택
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements 파일 복사 및 설치
COPY datapipeline/upload_requirements.txt .
RUN pip install -r upload_requirements.txt

# src 전체 복사
COPY datapipeline/src /app/src

# 실행 커맨드 (원한다면 직접 실행 가능)
CMD ["python", "src/cloud/upload_script.py"]

ENV PYTHONPATH=/app
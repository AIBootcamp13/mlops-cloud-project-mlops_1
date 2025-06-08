# 베이스 이미지 선택
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements 파일 복사 및 설치
COPY ./upload_requirements.txt .
RUN pip install --no-cache-dir -r upload_requirements.txt

# src 전체 복사
COPY ./src /app/src

# 환경변수로 PYTHONPATH 지정
ENV PYTHONPATH=/app/src

# 실행 커맨드
CMD ["python", "cloud/upload_script.py"]
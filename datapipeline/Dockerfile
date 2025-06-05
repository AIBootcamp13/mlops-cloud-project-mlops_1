# 1. 베이스 이미지
FROM python:3.10-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 의존성 설치
COPY upload_requirements.txt .
RUN pip install --no-cache-dir -r upload_requirements.txt

# 4. 소스 코드 복사
COPY datapipeline/src/ ./src/

# 5. 실행 환경 설정 및 실행 명령
ENV PYTHONPATH=/app
CMD ["python", "src/cloud/upload_script.py"]
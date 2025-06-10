FROM python:3.10-slim

WORKDIR /app
COPY .env /app
COPY data/ /app/data/
COPY modeling/ /app/modeling/
COPY datapipeline/ /app/datapipeline/
COPY serving/ /app/serving/

COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    default-libmysqlclient-dev \
    pkg-config \
    git \
    curl && \
    apt-get clean

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV AIRFLOW__CORE__DAGS_FOLDER=/app/modeling/src/airflow
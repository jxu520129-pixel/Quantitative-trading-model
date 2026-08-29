ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim@sha256:a630a63cdb314e2d138a2fca3e375e319e8568346ffafac5b980f888630ac4f1
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt
COPY . .
RUN mkdir -p /app/data

EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

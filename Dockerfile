FROM python:3.11-slim

RUN mkdir -p /core
WORKDIR /core

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libpcre3 \
    libssl-dev \
    libffi-dev \
    libpcre3-dev

RUN apt-get install -y libglib2.0-0
RUN apt-get install -y gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY . /core/

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
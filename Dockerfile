FROM python:3.10

# Установка системной зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY ./src ./src

RUN pip install -r requirements.txt

CMD ["python", "./src/train.py"]

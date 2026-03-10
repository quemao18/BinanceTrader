FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_specific.txt ./
RUN pip install --upgrade pip && \
    if [ -f requirements_specific.txt ]; then \
        pip install -r requirements_specific.txt; \
    else \
        pip install -r requirements.txt; \
    fi

COPY . .

CMD ["python", "BinanceTrader.py"]
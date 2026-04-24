# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# Stage 2: Final runtime image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

RUN mkdir -p $CONTAINER_HOME/src/data && \
    wget -q -O $CONTAINER_HOME/src/data/chunk_search_index.pkl \
    https://github.com/ArielVilensky/laughDB/releases/download/v1-indices/chunk_search_index.pkl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ARG CACHEBUST=1
COPY src/ $CONTAINER_HOME/src/
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["python", "-m", "gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000", "--log-level", "debug", "--preload", "--threads", "2"]

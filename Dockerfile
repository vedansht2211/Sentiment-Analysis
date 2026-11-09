# start with a light weight python image
FROM python:3.13-slim

# set the working directory inside the container
WORKDIR /app

# copy the dependency list before copying the application code
COPY requirements.txt .

# avoid retaining pip's download cache in the image
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code (the model is downloaded from MLflow at startup)
COPY main.py .

# give the model download plenty of time (30 minutes)
ENV MLFLOW_HTTP_REQUEST_TIMEOUT=1800
ENV MLFLOW_HTTP_REQUEST_MAX_RETRIES=3

# document the port used by the FastAPI application
EXPOSE 8000

# start the api server when the container launches
# 0.0.0.0 allows connections through the container's network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
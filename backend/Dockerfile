FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++ make git ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel
COPY requirements.linux.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 5001
CMD ["python", "app.py"]

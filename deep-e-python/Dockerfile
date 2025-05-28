FROM ghcr.io/deepextension/deep_e/grpo-base:v0.1

WORKDIR /app/deep-e-python

RUN mkdir -p /models /datasets

VOLUME ["/models", "/datasets"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/deep-e-python

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

COPY requirements_cuda.txt .

RUN pip install --no-cache-dir -r requirements_cuda.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]

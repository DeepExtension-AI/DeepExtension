FROM jianweisoft.cn/scpe/grpo-base:latest

WORKDIR /app/deep-e-python

# 建议：提前升级 pip，避免老版本解析 +cuXXX 失败
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip

# 复制代码与依赖清单
COPY . /app/deep-e-python
COPY requirements_cuda.txt /app/deep-e-python/

# 让 pip 同时使用清华镜像(常规包) + PyTorch 官方仓库(GPU 轮子)
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /app/deep-e-python/requirements_cuda.txt

# 其余保持不变
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]

FROM jianweisoft.cn/scpe/grpo-base:latest

# 安装 Conda 依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nvidia-cuda-toolkit
# 下载并安装 Miniconda
# 使用 conda-linux-64 指定旧版本 Miniconda（含 Python 3.10）
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
# 将 Conda 加入 PATH
ENV PATH="/opt/conda/bin:$PATH"

# 让 Conda 能发现系统 Python 库（关键步骤！）
RUN echo "/usr/local/lib/python3.10/dist-packages" > /opt/conda/lib/python3.10/site-packages/system-packages.pth

# 验证安装
RUN conda --version && python -c "from flask import Flask; "

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

# 使用预构建的flux-base基础镜像
FROM jianweisoft.cn/deep_e/flux-base:latest

# 复制requirements.txt
COPY requirements.txt .

# 安装其他Python依赖（PyTorch和nunchaku已在基础镜像中安装）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码
COPY . .

# 启动命令
CMD ["python", "dynamic_flux_app.py"]

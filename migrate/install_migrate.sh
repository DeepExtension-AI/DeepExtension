#!/bin/bash

# 固定版本号
version="v4.18.3"

# 检测操作系统（只可能是 darwin 或 linux）
case "$(uname -s)" in
    Darwin*) os="darwin" ;;
    Linux*)  os="linux" ;;
    *)       echo "Unsupported OS"; exit 1 ;;
esac

# 检测架构（只可能是 amd64 或 arm64）
case "$(uname -m)" in
    x86_64*) arch="amd64" ;;
    aarch64*|arm64*) arch="arm64" ;;
    *)       echo "Unsupported architecture"; exit 1 ;;
esac

# 输出检测结果
echo "Detected OS: $os"
echo "Detected Arch: $arch"

# 构造下载链接
url="https://github.com/golang-migrate/migrate/releases/download/${version}/migrate.${os}-${arch}.tar.gz"

# 执行下载并解压到临时目录
temp_dir=$(mktemp -d)
echo "Downloading and extracting to ${temp_dir}..."
if curl -sSL "$url" | tar xvz -C "$temp_dir"; then
    # 查找解压出来的migrate二进制文件路径
    migrate_bin="${temp_dir}/migrate"

    if [ -f "${migrate_bin}" ]; then
        # 使用sudo移动二进制文件到/usr/local/bin/
        sudo mv "${migrate_bin}" /usr/local/bin/migrate

        # 确认是否成功移动
        if [ -f "/usr/local/bin/migrate" ]; then
            echo "✅ Successfully installed migrate to /usr/local/bin/"

            # 添加可执行权限
            sudo chmod +x /usr/local/bin/migrate

            echo "You can now run the 'migrate' command from anywhere."
        else
            echo "❌ Failed to move the binary to /usr/local/bin/. Please check permissions."
            exit 1
        fi
    else
        echo "❌ Failed to find the migrate binary in the downloaded archive."
        exit 1
    fi

    # 删除临时目录
    rm -rf "$temp_dir"
else
    echo "❌ Failed to download or extract the migrate binary."
    exit 1
fi
#!/bin/bash

# Fixed version number
version="v4.18.3"

# Detect operating system (supports darwin, linux, and windows)
case "$(uname -s)" in
    Darwin*) os="darwin" ;;
    Linux*)  os="linux" ;;
    MINGW*|MSYS*|CYGWIN*) os="windows" ;;
    *) echo "❌ Unsupported operating system"; exit 1 ;;
esac

# Detect architecture (only amd64 or arm64 possible)
case "$(uname -m)" in
    x86_64*) arch="amd64" ;;
    aarch64*|arm64*) arch="arm64" ;;
    *) echo "❌ Unsupported architecture"; exit 1 ;;
esac

# Output detection results
echo "Detected OS: $os"
echo "Detected Arch: $arch"

# Construct download URL
if [ "$os" = "windows" ]; then
    url="https://github.com/golang-migrate/migrate/releases/download/${version}/migrate.${os}-${arch}.zip"
else
    url="https://github.com/golang-migrate/migrate/releases/download/${version}/migrate.${os}-${arch}.tar.gz"
fi

# Download and extract to temporary directory
temp_dir=$(mktemp -d)
echo "Downloading and extracting to ${temp_dir}..."

if [ "$os" = "windows" ]; then
    # Windows download and extraction logic
    if curl -sSL "$url" -o "${temp_dir}/migrate.zip" && unzip -q "${temp_dir}/migrate.zip" -d "$temp_dir"; then
        migrate_bin="${temp_dir}/migrate.exe"
    else
        echo "❌ Failed to download or extract the migrate binary."
        exit 1
    fi
else
    # Linux/Mac download and extraction logic
    if curl -sSL "$url" | tar xvz -C "$temp_dir"; then
        migrate_bin="${temp_dir}/migrate"
    else
        echo "❌ Failed to download or extract the migrate binary."
        exit 1
    fi
fi

if [ -f "${migrate_bin}" ]; then
    if [ "$os" = "windows" ]; then
        # Windows installation logic
        install_dir="/usr/local/bin"
        if [ ! -d "$install_dir" ]; then
            mkdir -p "$install_dir"
        fi
        cp "${migrate_bin}" "${install_dir}/migrate.exe"
        if [ -f "${install_dir}/migrate.exe" ]; then
            echo "✅ Successfully installed migrate to ${install_dir}/"
            chmod +x "${install_dir}/migrate.exe"
            echo "You can now run the 'migrate' command from anywhere."
        else
            echo "❌ Failed to copy the binary to ${install_dir}/. Please check permissions."
            exit 1
        fi
    else
        # Linux/Mac installation logic
        sudo mv "${migrate_bin}" /usr/local/bin/migrate
        if [ -f "/usr/local/bin/migrate" ]; then
            echo "✅ Successfully installed migrate to /usr/local/bin/"
            sudo chmod +x /usr/local/bin/migrate
            echo "You can now run the 'migrate' command from anywhere."
        else
            echo "❌ Failed to move the binary to /usr/local/bin/. Please check permissions."
            exit 1
        fi
    fi
else
    echo "❌ Failed to find the migrate binary in the downloaded archive."
    exit 1
fi

# Clean up temporary directory
rm -rf "$temp_dir"
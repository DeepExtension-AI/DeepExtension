# 删除show_diff.sh

# 删除set_env_variable.sh


#!/bin/bash

# Script to:
# 1. Merge dev.env into .env (add new fields or update existing ones)
# 2. Delete a specified directory

# Check if files exist
if [ ! -f "./environment/devpre.env" ]; then
    echo "Error: devpre.env file not found!"
    exit 1
fi

# Create .env if it doesn't exist
touch .env

# Backup original .env
cp .env .env.bak

# Process each line in dev.env
while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^# ]] || [[ -z "$line" ]]; then
        continue
    fi

    # Extract key and value
    key=$(echo "$line" | cut -d= -f1)
    value=$(echo "$line" | cut -d= -f2-)

    # Check if key exists in .env
    if grep -q "^$key=" .env; then
        if [ "$SYSTEM" = "Darwin" ];then
           sed -i '' "s/^$key=.*/$line/" ".env"
        else if [ "$SYSTEM" = "Linux" ];then
            sed -i "s/^$key=.*/$line/" ".env"
          fi
        fi
    else
        # Add new key
        echo "$line" >> .env
    fi
done < "./environment/devpre.env"

echo "Successfully merged dev.env into .env"

# Delete directory if specified
dir_to_delete="./environment"
if [ -d "./environment" ]; then
    rm -rf "$dir_to_delete"
    echo "Successfully deleted directory: $dir_to_delete"
else
    echo "Warning: Directory $dir_to_delete does not exist"
fi

file_path=".env.bak"  # 替换为你的文件路径

if [ -f "$file_path" ]; then
    echo "${file_path}存在，正在删除..."
    rm -f "$file_path"
    echo "${file_path}已删除"
else
    echo "${file_path}不存在，无需删除"
fi
file_path="image.env"  # 替换为你的文件路径

if [ -f "$file_path" ]; then
    echo "${file_path}存在，正在删除..."
    rm -f "$file_path"
    echo "${file_path}已删除"
else
    echo "${file_path}不存在，无需删除"
fi

file_path="set_env_variable.sh"  # 替换为你的文件路径

if [ -f "$file_path" ]; then
    echo "${file_path}存在，正在删除..."
    rm -f "$file_path"
    echo "${file_path}已删除"
else
    echo "${file_path}不存在，无需删除"
fi
file_path="show_diff.sh"  # 替换为你的文件路径

if [ -f "$file_path" ]; then
    echo "${file_path}存在，正在删除..."
    rm -f "$file_path"
    echo "${file_path}已删除"
else
    echo "${file_path}不存在，无需删除"
fi


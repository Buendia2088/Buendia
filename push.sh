#!/bin/bash
# 用法: ./git_init_commit_push.sh "提交说明"

# 判断是否有参数
if [ -z "$1" ]; then
  echo "❌ 请输入提交信息，例如:"
  echo "./git_init_commit_push.sh \"initial commit\""
  exit 1
fi

COMMIT_MSG="$1"
USER_NAME=${2:-"Buendia2088"}
USER_EMAIL=${3:-"cy1656736387@gmail.com"}

# 初始化仓库
git init

# 绑定远程（如果没有的话，可以修改为你自己的 repo 地址）
# ⚠️ 如果已经有 remote，这里会失败，可以忽略
git remote add origin https://github.com/Buendia2088/Buendia.git

# 配置用户身份（仅当前仓库，不影响全局）
git config user.name "$USER_NAME"
git config user.email "$USER_EMAIL"

# 添加所有文件
git add -A

# 提交
git commit -m "$COMMIT_MSG"

# 确保分支是 main，如果没有则新建
git branch -M main

# 推送到远程
git push -u origin main

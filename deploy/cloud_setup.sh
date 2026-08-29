#!/usr/bin/env bash
# 腾讯云轻量应用服务器（2核2G）部署脚本
# 用法：把整个项目上传到服务器后，在项目根目录执行：
#   bash deploy/cloud_setup.sh
# 幂等可重复执行；适合 Docker CE 应用镜像或已装 Docker 的 Ubuntu。
set -euo pipefail

echo "==> [1/6] 配置 2G swap（防止每日 16:10 模型重训 OOM）"
if ! swapon --show 2>/dev/null | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
swapon --show

echo "==> [2/6] 安装 Docker 与 Compose 插件（若未安装）"
if ! command -v docker >/dev/null 2>&1; then
  echo "未检测到 Docker，正在安装（官方脚本，约 2~3 分钟）..."
  if ! command -v curl >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y curl; fi
  curl -fsSL https://get.docker.com | sudo sh
fi
# 把当前用户加入 docker 组（本次脚本仍用 sudo，重登后免 sudo）
sudo usermod -aG docker "$USER" 2>/dev/null || true
sudo docker --version
if ! sudo docker compose version >/dev/null 2>&1; then
  echo "缺少 docker compose 插件，正在安装..."
  sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi
sudo docker compose version

echo "==> [3/6] 调整重训采样（2G 内存：300 -> 120）"
sed -i 's/^  retrain_sample_symbols:.*/  retrain_sample_symbols: 120/' config/default.yaml
grep -n "retrain_sample_symbols" config/default.yaml

echo "==> [4/6] 初始化 .env"
if [ ! -f .env ]; then
  cp .env.example .env
fi
if grep -qE 'DASHBOARD_PASSWORD=(请在本机修改为强密码|)$' .env; then
  echo "=============================================================="
  echo "请先设置看板管理员密码：编辑 .env 文件，把 DASHBOARD_PASSWORD 改成强密码"
  echo "  vi .env"
  echo "改完后重新执行： bash deploy/cloud_setup.sh"
  echo "=============================================================="
  exit 1
fi

echo "==> [5/6] 构建并启动容器"
sudo docker compose up -d --build

echo "==> [6/6] 启动状态"
sudo docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "=============================================================="
echo "部署完成！看板地址： http://<你的公网IP>:8501 （登录密码为 .env 中的 DASHBOARD_PASSWORD）"
echo "查看调度日志： sudo docker compose logs -f scheduler"
echo "补拉真实行情（约 2 小时）： sudo docker compose exec -T scheduler python -m ashare_quant.cli update-data"
echo "提示：重登 SSH 后 docker 命令可免 sudo（已加入 docker 组）"
echo "=============================================================="

#!/bin/bash
set -e

sudo apt update
sudo apt install -y docker.io

sudo systemctl enable docker
sudo systemctl start docker

sudo usermod -aG docker "$USER"

echo "Docker installed. Log out and log back in for group changes to apply."

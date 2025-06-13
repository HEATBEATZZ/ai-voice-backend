#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    portaudio19-dev \
    python3-pyaudio \
    build-essential \
    python3-dev

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt 

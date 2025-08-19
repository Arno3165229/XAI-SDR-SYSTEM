#!/bin/bash

# Automated script to install dependencies, build Docker image, and run the container

# Check for prerequisites
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Install Python dependencies for local development (optional)
pip install -r requirements.txt

# Set API key if not set
if [ -z "$XAI_API_KEY" ]; then
    read -p "Enter your XAI_API_KEY: " XAI_API_KEY
    export XAI_API_KEY
fi

# Build and run Docker
docker-compose up --build
#!/bin/bash
# Build and push Docker images for LingBot-VA environments

# Configuration
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-your-dockerhub-username}"
IMAGE_NAME_BASE="${DOCKER_HUB_USERNAME}/lingbot-va"
VERSION="20260316"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LingBot-VA Docker Build & Push Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Docker Hub username
if [ "$DOCKER_HUB_USERNAME" = "your-dockerhub-username" ]; then
    echo -e "${RED}Error: Please set DOCKER_HUB_USERNAME${NC}"
    echo "Usage: DOCKER_HUB_USERNAME=yourname ./build.sh"
    exit 1
fi

# Login to Docker Hub
echo -e "\n${YELLOW}Login to Docker Hub...${NC}"
docker login

# Build lingbot-va image
echo -e "\n${YELLOW}Building lingbot-va image...${NC}"
cd /home/syr/code/lingbot-va
docker build -f docker/Dockerfile.lingbot-va -t ${IMAGE_NAME_BASE}:${VERSION} -t ${IMAGE_NAME_BASE}:latest .

# Push lingbot-va image
echo -e "\n${YELLOW}Pushing lingbot-va image...${NC}"
docker push ${IMAGE_NAME_BASE}:${VERSION}
docker push ${IMAGE_NAME_BASE}:latest

# Build robotwin image
echo -e "\n${YELLOW}Building robotwin image...${NC}"
docker build -f docker/Dockerfile.robotwin -t ${IMAGE_NAME_BASE}-robotwin:${VERSION} -t ${IMAGE_NAME_BASE}-robotwin:latest .

# Push robotwin image
echo -e "\n${YELLOW}Pushing robotwin image...${NC}"
docker push ${IMAGE_NAME_BASE}-robotwin:${VERSION}
docker push ${IMAGE_NAME_BASE}-robotwin:latest

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build and push completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nImages available:"
echo "  - ${IMAGE_NAME_BASE}:${VERSION}"
echo "  - ${IMAGE_NAME_BASE}:latest"
echo "  - ${IMAGE_NAME_BASE}-robotwin:${VERSION}"
echo "  - ${IMAGE_NAME_BASE}-robotwin:latest"

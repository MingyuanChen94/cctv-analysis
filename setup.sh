#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up CCTV Analysis System...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Found Python version: ${python_version}${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment created successfully${NC}"

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to upgrade pip${NC}"
    exit 1
fi
echo -e "${GREEN}Pip upgraded successfully${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}Dependencies installed successfully${NC}"

# Install development dependencies
echo -e "\n${YELLOW}Installing development dependencies...${NC}"
pip install -e ".[dev]"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install development dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}Development dependencies installed successfully${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p output/videos output/visualizations output/reports
mkdir -p logs
mkdir -p temp

# Setup pre-commit hooks
echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
pre-commit install
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to setup pre-commit hooks${NC}"
    exit 1
fi
echo -e "${GREEN}Pre-commit hooks installed successfully${NC}"

echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "\nTo activate the virtual environment, run: ${YELLOW}source venv/bin/activate${NC}"
echo -e "To deactivate, run: ${YELLOW}deactivate${NC}"

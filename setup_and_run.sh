#!/bin/bash

# Step 1: Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate the virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirement.txt

# Step 4: Run the main script
echo "Starting the main script..."
python3 main.py

# Step 5: Deactivate the virtual environment
deactivate


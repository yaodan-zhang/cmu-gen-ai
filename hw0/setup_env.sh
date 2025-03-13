#!/bin/bash

# Set the virtual environment name
VENV_NAME="myenv"

echo "Creating virtual environment: $VENV_NAME"
python3 -m venv $VENV_NAME

echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r handout/requirements.txt

echo "Installation complete! Virtual environment is now active."
echo "To deactivate, type: deactivate"

@echo off
echo Setting up CCTV Analysis System...

:: Check Python version
python --version
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

:: Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    exit /b 1
)
echo Virtual environment created successfully

:: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)
echo Virtual environment activated

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Error: Failed to upgrade pip
    exit /b 1
)
echo Pip upgraded successfully

:: Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed successfully

:: Install development dependencies
echo.
echo Installing development dependencies...
pip install -e .[dev]
if errorlevel 1 (
    echo Error: Failed to install development dependencies
    exit /b 1
)
echo Development dependencies installed successfully

:: Create necessary directories
echo.
echo Creating project directories...
mkdir data\raw data\processed
mkdir models
mkdir output\videos output\visualizations output\reports
mkdir logs
mkdir temp

:: Setup pre-commit hooks
echo.
echo Setting up pre-commit hooks...
pre-commit install
if errorlevel 1 (
    echo Error: Failed to setup pre-commit hooks
    exit /b 1
)
echo Pre-commit hooks installed successfully

echo.
echo Setup completed successfully!
echo.
echo To activate the virtual environment, run: venv\Scripts\activate
echo To deactivate, run: deactivate

pause

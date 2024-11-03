# CCTV Analysis Package - Project Structure

## Overview

The CCTV Analysis Package is organized as follows:

```plaintext
cctv-analysis/
│
├── src/                              # Source code directory
│   └── cctv_analysis/               # Main package directory
│       ├── __init__.py             # Package initialization
│       ├── camera_processor.py     # Main video processing module
│       ├── person_detector.py      # Person detection implementation
│       ├── person_tracker.py       # Person tracking implementation
│       ├── demographic_analyzer.py # Demographic analysis module
│       │
│       └── utils/                  # Utilities package
│           ├── __init__.py        # Utilities initialization
│           ├── logging_utils.py   # Logging configuration
│           ├── config_utils.py    # Configuration management
│           ├── data_utils.py      # Data processing utilities
│           ├── video_utils.py     # Video handling utilities
│           ├── visualization_utils.py  # Visualization tools
│           ├── metrics_utils.py   # Metrics calculation
│           └── report_utils.py    # Report generation
│
├── config/                          # Configuration directory
│   ├── default_config.yaml         # Default configuration
│   ├── production_config.yaml      # Production settings
│   └── test_config.yaml           # Testing configuration
│
├── data/                           # Data directory
│   ├── raw/                       # Raw video files
│   └── processed/                 # Processed data
│
├── models/                         # Model weights and configs
│   ├── detection/                 # Detection models
│   ├── tracking/                  # Tracking models
│   └── demographic/               # Demographic analysis models
│
├── output/                         # Output directory
│   ├── videos/                    # Processed videos
│   ├── visualizations/            # Generated visualizations
│   └── reports/                   # Analysis reports
│
├── templates/                      # Report templates
│   ├── report_template.html       # HTML report template
│   └── report_template.md         # Markdown report template
│
├── tests/                          # Test directory
│   ├── __init__.py               # Test initialization
│   ├── test_camera_processor.py  # Camera processor tests
│   ├── test_person_detector.py   # Detector tests
│   ├── test_person_tracker.py    # Tracker tests
│   └── test_utils/               # Utility tests
│
├── logs/                           # Log files directory
│
├── docs/                           # Documentation
│   ├── api.md                    # API documentation
│   ├── installation.md           # Installation guide
│   └── usage.md                  # Usage guide
│
├── scripts/                        # Utility scripts
│   ├── setup.sh                  # Unix setup script
│   └── setup.bat                 # Windows setup script
│
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
├── .pre-commit-config.yaml        # Pre-commit hook configuration
├── pyproject.toml                 # Project configuration
├── setup.py                       # Package setup file
├── requirements.txt               # Project dependencies
├── README.md                      # Project readme
└── STRUCTURE.md                   # This file

```

## Component Details

### Source Code (`src/`)

- `camera_processor.py`: Main module for processing CCTV footage
- `person_detector.py`: Implements person detection using various backends
- `person_tracker.py`: Handles person tracking and re-identification
- `demographic_analyzer.py`: Analyzes demographic attributes
- `utils/`: Package containing utility modules for various functionalities

### Configuration (`config/`)

- YAML files containing configuration for different environments
- Includes settings for models, processing parameters, and system options

### Data Management (`data/`)

- `raw/`: Original CCTV footage and input files
- `processed/`: Processed data and intermediate results

### Models (`models/`)

- Pre-trained models and weights
- Organized by functionality (detection, tracking, demographics)

### Output (`output/`)

- `videos/`: Processed and annotated videos
- `visualizations/`: Generated plots and visualizations
- `reports/`: Analysis reports in various formats

### Templates (`templates/`)

- HTML and Markdown templates for report generation
- Configurable layouts and styling

### Tests (`tests/`)

- Unit tests for all components
- Integration tests for end-to-end functionality
- Test utilities and mock data

### Documentation (`docs/`)

- API documentation
- Installation instructions
- Usage guides and examples

### Scripts (`scripts/`)

- Setup and automation scripts
- Environment configuration helpers

### Configuration Files

- `.pre-commit-config.yaml`: Code quality checks configuration
- `pyproject.toml`: Project and tool settings
- `setup.py`: Package installation configuration
- `requirements.txt`: Project dependencies

## Environment Setup

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Unix
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Usage

The primary package can be imported as:

```python
from cctv_analysis import CameraProcessor
from cctv_analysis.utils import (
    DataManager,
    VideoProcessor,
    VisualizationManager
)
```

## Directory Management

- Log files are automatically saved to `logs/`
- Processed videos are saved to `output/videos/`
- Reports are generated in `output/reports/`
- Temporary files are stored in `temp/` (git-ignored)

## Best Practices

1. Always use the virtual environment
2. Follow the existing directory structure
3. Run tests before committing changes
4. Update documentation as needed
5. Use configuration files for settings
6. Follow code style guidelines (enforced by pre-commit hooks)

## Notes

- The `temp/` directory is ignored by git and used for temporary files
- Log files are rotated automatically
- Configuration files should not contain sensitive information
- Model weights should be downloaded separately

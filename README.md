# CCTV Analysis System

A comprehensive system for analyzing CCTV footage to track individuals across multiple cameras and perform demographic analysis.

## Features

- Multi-camera person tracking
- Person re-identification between cameras
- Demographic analysis (age and gender)
- Timestamp-based movement analysis
- Privacy-focused data handling
- Configurable processing pipeline

## Project Structure

```plaintext
cctv-analysis/
│
├── src/                                # Source code directory
│   └── cctv_analysis/                  # Main package directory
│       ├── __init__.py                 # Package initialization
│       ├── detector.py                 # Person detection module
│       ├── tracker.py                  # Person tracking module
│       ├── matcher.py                  # Person matching across cameras
│       ├── reid.py                     # Re-identification module
│       ├── demographics.py             # Demographic analysis module
│       │
│       └── utils/                      # Utility modules
│           ├── __init__.py
│           ├── config.py               # Configuration handling
│           ├── visualization.py        # Visualization utilities
│           ├── metrics.py              # Analysis metrics
│           └── types.py                # Data type definitions
│
├── data/                               # Data directory
│   └── videos/                         # Video files
│
├── models/                             # Model weights
│   ├── README.md                       # Models documentation
│   ├── detector/                       # YOLOX weights
│   ├── reid/                           # ReID model weights
│
├── examples/                           # Example code
│   ├── basic_tracking.ipynb            # Basic tracking example
│   ├── multi_camera.ipynb              # Multi-camera analysis
│   └── demographic_analysis.ipynb      # Demographic analysis example
│
├── tests/                              # Test files
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_matcher.py
│   └── test_demographics.py
│
├── scripts/                            # Utility scripts
│   ├── setup.sh                        # Setup script
│   ├── download_models.py              # Model download script
│   └── verify_models.py                # Model verification
│
├── configs/                            # Configuration files
│   └── models.yaml                     # Model URLs and checksums
│
├── pyproject.toml                      # Build configuration
├── setup.cfg                           # Package metadata
├── setup.py                            # Setup script
├── requirements.txt                    # pip requirements
├── environment.yml                     # Conda environment file
│
├── LICENSE                             # License file
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

## Prerequisites

- Anaconda or Miniconda
- CUDA-capable GPU (recommended)
- Git

## Installation Guide for CCTV Analysis

### Prerequisties

1. **System Requirements**:

   - Python 3.8
   - CUDA-capable GPU (recommended)
   - CUDA Toolkit 11.5 or higher

2. **Required Software**:

   - Anaconda or Miniconda ([Download here](https://docs.conda.io/en/latest/miniconda.html))
   - Git ([Download here](https://git-scm.com/downloads))

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MingyuanChen94/cctv-analysis.git
    cd cctv-analysis
    ```

2. Create and activate the conda environment:

    ```bash
    # Create environment
    conda env create --file=environment.yml # GPU
    conda env create --file=environment_cpu.yml # CPU

    # Activate environment
    conda activate cctv-analysis
    ```

3. Download models:

    ```bash
    # Download to default location (./models/)
    python scripts/download_models.py

    # Download to specific directory
    python scripts/download_models.py --models-dir /path/to/models

    # Download specific model
    python scripts/download_models.py --models reid
    ```

4. Verify the models:

    ```bash
    # Verify all models
    python scripts/verify_models.py

    # Verify specific model
    python scripts/verify_models.py --model reid

    # Verify models in custom directory
    python scripts/verify_models.py --models-dir /path/to/models

    # Show detailed information
    python scripts/verify_models.py --verbose
    ```

5. Verify the installation:

    ```python
    import cctv_analysis
    print(cctv_analysis.__version__)
    ```

## Usage

1. Configure your camera settings in `config/config.yaml`

2. Run the analysis

    ```python
    import os
    from cctv_analysis.camera_processor import process_surveillance_footage

    results = process_surveillance_footage(
        camera1_path=os.path.join("data","raw","camera1"),
        camera2_path=os.path.join("data","raw","camera2")
    )
    ```

3. View the results

    ```python
    print(results.head())
    ```

## Privacy Considerations

This system is designed with privacy in mind

- All processed data is anonymized
- No raw footage is stored
- Configurable data retention policies
- Compliance with GDPR and other privacy regulations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Future Improvements

- [ ] Add support for real-time processing
- [ ] Implement multiple camera support (>2)
- [ ] Add GUI for parameter tuning
- [ ] Improve matching accuracy
- [ ] Add export to various formats

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cctv_analysis_2024,
  author = {Mingyuan Chen},
  title = {CCTV Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/MingyuanChen94/cctv-analysis}
}
```

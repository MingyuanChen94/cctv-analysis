# CCTV Analysis Examples

This directory contains example scripts demonstrating various use cases of the CCTV Analysis system.

## Examples List

1. `basic_analysis.py` - Basic video analysis with person detection and tracking
2. `multi_camera_analysis.py` - Analysis of multiple camera feeds with person re-identification
3. `demographic_analysis.py` - Detailed demographic analysis of detected individuals
4. `real_time_processing.py` - Real-time video processing example
5. `visualization_examples.py` - Different visualization options for analysis results
6. `report_generation.py` - Custom report generation examples

## Usage

Make sure you have installed the package in development mode:

```bash
pip install -e .
```

Then run any example script:

```bash
python examples/basic_analysis.py
python examples/multi_camera_analysis.py
# etc.
```

## Data

Example scripts expect input files in the `data/` directory:

- Video files in `data/raw/`
- Processed data in `data/processed/`

## Output

Results will be saved in the `output/` directory:

- Processed videos in `output/videos/`
- Visualizations in `output/visualizations/`
- Reports in `output/reports/`

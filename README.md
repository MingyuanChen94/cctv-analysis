# CCTV Analysis System

A comprehensive system for analyzing CCTV footage to track individuals across multiple cameras and perform demographic analysis.

## Features

- Multi-camera person tracking
- Person re-identification between cameras
- Demographic analysis (age and gender)
- Timestamp-based movement analysis
- Privacy-focused data handling
- Configurable processing pipeline

## Requirements

- Python 3.8+
- OpenCV 4.5+
- DeepFace
- NumPy
- Pandas
- PyYAML

## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/cctv-analysis.git
cd cctv-analysis
```

2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your camera settings in `config/config.yaml`

2. Run the analysis

```python
import os
from cctv_analysis.camera_processor import process_surveillance_footage

results = process_surveillance_footage(
    camera1_path=os.path.join("data","raw","camera1"),
    camera2_path=os.path.join("data","raw","camera1")
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project Structure

- [[structure|Project Structure]]

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

# Model Weights

This directory contains the pre-trained model weights used by the CCTV Analysis package.

## Required Models

1. **YOLOX-L** (Person Detection)
   - File: `detector/yolox_l.pth`
   - Size: 155.4 MB
   - Source: [YOLOX GitHub](https://github.com/Megvii-BaseDetection/YOLOX)

2. **OSNet** (Person Re-identification)
   - File: `reid/osnet_x1_0.pth`
   - Size: 73.2 MB
   - Source: [Deep Person ReID](https://github.com/KaiyangZhou/deep-person-reid)

## Downloading the Models

### Option 1: Using the Download Script

```bash
# Download all models
python scripts/download_models.py

# Download specific models
python scripts/download_models.py --models detector reid

# Specify custom directory
python scripts/download_models.py --models-dir /path/to/models
```

### Option 2: Manual Download

If you prefer to download the models manually:

1. Create the required directories:

    ```bash
    mkdir -p models/detector models/reid
    ```

2. Download the models from their respective sources:

   - YOLOX-L: [Download Link](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)
   - OSNet: [Download Link](https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA)

3. Place the downloaded files in their respective directories:

   - `models/detector/yolox_l.pth`
   - `models/reid/osnet_x1_0.pth`

## Verifying Downloads

You can verify the integrity of downloaded models using:

```bash
python scripts/download_models.py --verify
```

## Model Directory Structure

```plaintext
models/
├── detector/
│   └── yolox_l.pth
├── reid/
│   └── osnet_x1_0.pth
└── README.md
```

## Notes

- Models are not included in the Git repository due to their size
- Make sure you have sufficient disk space before downloading
- All models are downloaded with version checking and MD5 verification
- The script will skip already downloaded models if their MD5 hash matches

# SmartWear Advisor Desktop

SmartWear Advisor Desktop is a Python Tkinter application that combines webcam-based face recognition, accessory detection, weather data, and recommendation logic to suggest whether a user should wear a mask or glasses.

## Features

- Desktop UI built with Tkinter
- Local webcam-based face recognition workflow
- Mask and glasses detection with local TensorFlow Lite models
- Weather-aware recommendation engine
- City-based weather lookup
- Registered user encoding data stored locally
- Model training and dataset download scripts
- Separate backend, core detection, AI model, and desktop UI modules

## Tech Stack

- Python
- Tkinter
- OpenCV
- face-recognition and dlib
- TensorFlow / TensorFlow Lite
- NumPy and Pillow
- requests

## Project Structure

```text
mainapp.py                  Application launcher
desktop_app/                Tkinter desktop interface
core/                       Face and accessory detection engines
backend/                    Weather and recommendation logic
ai_model/                   Dataset download, training, and detection scripts
models/                     Mask and glasses model files
data/                       Registered user metadata and face encodings
requirements.txt            Python dependencies
runtime.txt                 Runtime version metadata
```

## Getting Started

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Some dependencies, especially `dlib` and `face-recognition`, may require platform-specific build tools.

### 3. Run the desktop app

```powershell
python mainapp.py
```

## Model Workflow

The repository includes local model files under `models/`. If model files are missing or you want to retrain them:

```powershell
python ai_model\download_datasets.py
python ai_model\train_model.py
python mainapp.py
```

## Recommendation Logic

The recommendation engine considers:

- Current temperature
- Weather condition
- Whether a mask is detected
- Whether glasses are detected

It then returns practical suggestions, such as wearing a mask in cold or polluted weather or considering sunglasses in clear conditions.

## Notes

This project uses local camera input and local model files. Make sure your webcam permissions are enabled before running the app.

## License

This repository is licensed under the GPL-3.0 license. See `LICENSE` for details.

# SmartWear Advisor Desktop

Desktop Tkinter version of SmartWear Advisor with:
- live face recognition from the local webcam
- mask and glasses detection with local TFLite models
- weather-aware recommendations for the selected city
- a desktop layout aligned with the browser version

If the model files are missing from `models/`:

1. run `ai_model/download_datasets.py`
2. run `ai_model/train_model.py`
3. run `mainapp.py`

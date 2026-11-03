# 🧠 NeuroVision-AI

End-to-end deep learning MRI tumor classification system. Upload a brain MRI slice and get a predicted tumor type (meningioma, glioma or pituitary), a confidence score and a short report, through a Flask web app.

> **Medical disclaimer:** This project is for education and research only. It is not a medical device and cannot diagnose any condition. Always consult a qualified doctor.

## Results

The model is a ResNet50 (ImageNet pretrained) fine-tuned with a custom classifier head, trained on the Jun Cheng brain tumor dataset (3,064 T1-weighted contrast-enhanced MRI slices from 233 patients).

The data was split **by patient** (70/15/15), so no patient appears in more than one of train, validation and test. This avoids the inflated accuracy that comes from splitting by image.

| Metric | Result |
|---|---|
| Test accuracy (unseen patients) | **93.4%** |
| Pituitary F1 | 0.997 |
| Glioma F1 | 0.928 |
| Meningioma F1 | 0.826 |

Meningioma and glioma are the classes the model confuses most often.

## Limitations

- It always predicts one of three tumor types. It has no "no tumor" class.
- It can give a confident answer for images that are not suitable brain MRI scans.
- It was trained on one dataset and one scan type, so it may perform worse on scans from other machines or hospitals.
- Confidence values are often very high, even for wrong predictions, so they should not be read as accuracy.

## Run locally

1. Install Python 3.11 or newer and clone this repo.
2. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install flask pillow
   ```

3. Download `bt_resnet50_model.pt` (see **Model file** below) and place it in the `models/` folder.
4. Start the app and open http://127.0.0.1:5000

   ```bash
   python app.py
   ```

Sample images to try are in `Brain-Tumor-Test-Images/`.

## Model file

The trained weights (about 128 MB) are too large for the repository. Download `bt_resnet50_model.pt` from the **Releases** page and put it in `models/`.

## Tech stack

Python, PyTorch, Torchvision, Flask, Pillow, Bootstrap.

## Credits

- Dataset: Jun Cheng, *Brain tumor dataset*, Figshare, DOI 10.6084/m9.figshare.1512427.
- Based on the original project [neuralBlack](https://github.com/aksh-ai/neuralBlack) by Akshay Kumaar M (Apache License 2.0). The model was retrained with a patient-level split, and the app code and web interface were reworked.

## Author

Indira
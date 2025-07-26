---
title: Vehicle Classification - README
---

\# 🚗 Vehicle Classification using EfficientNet-B0

This repository contains the full pipeline for a deep learning project
on multi-class vehicle classification using real-world image data. The
solution uses \*\*EfficientNet-B0\*\*, fine-tuned with domain-specific
augmentations and transfer learning. The final model achieves high
performance and is exported in \*\*ONNX\*\* format for deployment.

\-\--

\## 📁 Project Structure

\`\`\`  
vehicle-classification/  
├── Vehicle_detection\_(2) (1).ipynb \# Main notebook (cleaning,
training, evaluation)  
├── vehicles_classifier.onnx \# Final model in ONNX format  
├── classes.txt \# List of class names in prediction order  
├── Vehicle Classification Report.docx# Final detailed report  
├── /vehicle_dataset/ \# Dataset: train/val/test  
└── README.md \# Instructions (this file)  
\`\`\`

\-\--

\## ✅ Features

\- ✔️ Image cleaning, duplicate removal (manual + assisted)  
- ✔️ Stratified dataset split  
- ✔️ Data augmentation with \*\*Albumentations\*\*  
- ✔️ Transfer learning using \*\*EfficientNet-B0\*\*  
- ✔️ Model evaluation using accuracy, precision, recall, F1-score, mAP  
- ✔️ Exported model to \*\*ONNX\*\* for inference portability  
- ✔️ Comparative performance vs. ResNet18 and Custom CNN

\-\--

\## 🔧 Setup Instructions

\### 1. Clone the Repo

\`\`\`bash  
git clone https://github.com/your-username/vehicle-classification.git  
cd vehicle-classification  
\`\`\`

\### 2. Create Virtual Environment

\`\`\`bash  
python -m venv venv  
source venv/bin/activate \# On Windows: venv\Scripts\activate  
\`\`\`

\### 3. Install Dependencies

\`\`\`bash  
pip install -r requirements.txt  
\`\`\`

\*\*Key libraries used:\*\*

\- \`torch\`, \`torchvision\`  
- \`onnx\`, \`onnxruntime\`  
- \`albumentations\`, \`opencv-python\`  
- \`scikit-learn\`, \`matplotlib\`, \`pandas\`, \`numpy\`

\-\--

\## 📂 Dataset Structure

Expected structure inside \`/vehicle_dataset/\`:

\`\`\`  
vehicle_dataset/  
├── train/  
│ ├── bus/  
│ ├── car/  
│ └── \...  
├── val/  
│ └── (same as train/)  
└── test/  
└── (optional, for final evaluation)  
\`\`\`

\> \*\*Note\*\*: Dataset must be manually placed in this folder. For
privacy, it is not shared here.

\-\--

\## ✅ Google Colab Execution Guide

\### 🔼 Upload and Organize Your Files

1\. Upload the following to your \*\*Google Drive\*\*:  
- Dataset folder (e.g. \`vehicle_dataset/\`)  
- \`Vehicle_detection\_(2) (1).ipynb\`  
- \`vehicles_classifier.onnx\`  
- \`sample_classe.txt\`  
- \`verify_model.py\`  
- Optional: Add a subfolder \`verification/verification_images/\`
containing sample images like \`car.png\`, \`van.png\`, etc.

2\. Create this folder structure in Drive:

\`\`\`  
MyDrive/  
├── vehicle_dataset/  
│ ├── train/  
│ ├── val/  
├── verification/  
│ ├── vehicles_classifier.onnx  
│ ├── sample_classe.txt  
│ └── verification_images/  
│ ├── car.png  
│ ├── van.png  
│ └── autorickshaw.png  
\`\`\`

\-\--

\### ▶️ Running Notebook in Colab

1\. Right-click on the notebook file (\`Vehicle_detection\_(2)
(1).ipynb\`) in Drive.  
2. Select \*\*\"Open with \> Google Colab\"\*\*.  
3. At the top of the notebook, add the following to mount Drive:

\`\`\`python  
from google.colab import drive  
drive.mount(\'/content/drive\')  
\`\`\`

4\. Update dataset and model paths in the notebook cells, for example:

\`\`\`python  
dataset_path = \"/content/drive/MyDrive/vehicle_dataset\"  
model_save_path =
\"/content/drive/MyDrive/verification/vehicles_classifier.onnx\"  
\`\`\`

5\. Run all cells sequentially:  
- Data Cleaning  
- Preprocessing  
- Training  
- Evaluation  
- Model Export

\-\--

\## ✅ Model Verification: \`verify_model.py\`

This script verifies the exported ONNX model on test images.

\### ▶️ Steps to Run in Google Colab

1\. Upload and open \`verify_model.py\` in Colab.  
2. Make sure these files exist in
\`/content/drive/MyDrive/verification/\`:  
- \`vehicles_classifier.onnx\`  
- \`sample_classe.txt\`  
- Folder \`verification_images/\` with sample images

3\. Add this cell at the top:

\`\`\`python  
from google.colab import drive  
drive.mount(\'/content/drive\')  
\`\`\`

4\. Then run the entire script. It will:  
- Load your ONNX model and check its validity  
- Preprocess the test images  
- Make predictions  
- Annotate and save results in \`/runs/pred/\`

✅ Example output:  
\`\`\`  
✅ Prediction for car.png: car (0.91)  
Saved to ./runs/pred/pred_car.png  
\`\`\`

\-\--

\## 📦 Output Artifacts

\- \`runs/org/\`: Original verification images  
- \`runs/pred/\`: Annotated prediction images  
- \`vehicles_classifier.onnx\`: Trained model  
- \`Vehicle Classification Report.docx\`: Final analysis and results

\-\--

\## 🔮 Future Improvements

\- Ensemble learning for better generalization  
- YOLOv8 / ViT backbone experimentation  
- AutoML with Keras Tuner or Optuna  
- CleanLab integration for label correction  
- Deployment using Flask, Gradio, or TensorRT

\-\--

\## 🧑‍💻 Contact

For questions or collaborations, feel free to open an issue or reach
out.

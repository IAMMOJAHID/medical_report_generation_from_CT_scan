# 🧠 CT Scan Report Generation using Swin-UNETR and BLIP

This repository contains code to generate textual medical reports from 3D CT scans using a hybrid model that combines **Swin-UNETR** for vision encoding and **BLIP (Bootstrapped Language-Image Pretraining)** for conditional text generation. 

It is designed for multimodal medical tasks where the goal is to describe the medical content of a CT scan using natural language.

## 📌 Overview

- **Input**: 3D CT volumes (`.nii` format) + paired medical reports
- **Output**: Natural language reports that describe the CT scan
- **Vision Encoder**: Swin-UNETR (from MONAI)
- **Text Decoder**: BLIP (from HuggingFace Transformers)


## 📁 Project Structure

ReportGeneration/
│
├── SwinModel/
│ ├── main.py # Entry point for training
│ ├── train.py # Training loop
│ ├── dataloader.py # Custom dataset and dataloader
│ ├── SwinModel.py # Model definition (Swin-UNETR + BLIP)
│ └── utils.py # Padding, tokenization, etc.
│
├── Train/ # Folder containing training CT scans (.nii)
├── Validation/ # Folder containing validation CT scans
├── report_data.xlsx # Excel file mapping image names to reports
├── README.md # You're here
└── requirements.txt # Python dependencies
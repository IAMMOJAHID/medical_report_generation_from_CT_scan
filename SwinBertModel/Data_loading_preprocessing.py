import os
import glob
import pandas as pd
import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandFlipd, RandRotate90d, RandShiftIntensityd, ToTensord
)
from monai.data import Dataset, DataLoader

# Directories
TRAIN_DIR = "Train"
VAL_DIR = "Validation"
REPORT_FILE = "reports.xlsx"  # Your XLSX file containing reports

# Load Reports from Excel File
report_df = pd.read_excel(REPORT_FILE)  # Columns: ["filename", "report"]
report_dict = dict(zip(report_df["filename"], report_df["report"]))  # Convert to dictionary

# Get Image File Paths
train_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.nii.gz")))
val_images = sorted(glob.glob(os.path.join(VAL_DIR, "*.nii.gz")))

# Create Data Dictionaries
train_data = [{"image": img, "report": report_dict[os.path.basename(img)]} for img in train_images if os.path.basename(img) in report_dict]
val_data = [{"image": img, "report": report_dict[os.path.basename(img)]} for img in val_images if os.path.basename(img) in report_dict]

# Define Transforms
# Define transforms
train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image"], source_key="image"),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ToTensord(keys=["image"])
])

valid_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image"], source_key="image"),
    ToTensord(keys=["image"])
])

# Create Dataset and Dataloader
train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=valid_transforms)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

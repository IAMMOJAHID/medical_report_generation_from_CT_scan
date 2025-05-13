import os
import glob
import pandas as pd
import torch
# from torch.utils.data import Dataset, DataLoader
from monai.data import (
    DataLoader,
    CacheDataset,
)
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, SpatialPadd, Spacingd, ScaleIntensityRanged, CropForegroundd, RandFlipd, RandRotate90d, RandShiftIntensityd, RandSpatialCropSamplesd, ToTensord

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(288, 288, 224), method='symmetric'),
        RandSpatialCropSamplesd(
            keys=["image"],  # Only the image, no label required
            roi_size=(288, 288, 224),  # Size of the cropped region
            num_samples=1,  # Number of crops per image
            random_center=False,  # Randomly select the crop center
            random_size=False,  # Keep crop size fixed
        ),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ToTensord(keys=["image"])
    ])

def get_valid_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(288, 288, 224), method='symmetric'),
        RandSpatialCropSamplesd(
            keys=["image"],  # Only the image, no label required
            roi_size=(288, 288, 224),  # Size of the cropped region
            num_samples=1,  # Number of crops per image
            random_center=False,  # Randomly select the crop center
            random_size=False,  # Keep crop size fixed
        ),
        ToTensord(keys=["image"])
    ])


def get_dataloaders(train_dir, val_dir, report_file, batch_size=4):
    # Load Reports from Excel File
    report_df = pd.read_excel(report_file)  # Columns: ["filename", "report"]
    report_dict = dict(zip(report_df["AccessionNo"], report_df["Findings_EN"]))  # Convert to dictionary
    # print("report dict:", report_dict['amos_7383'])

    # Get Image File Paths
    train_images = sorted(glob.glob(os.path.join(train_dir, "*.nii.gz")))
    val_images = sorted(glob.glob(os.path.join(val_dir, "*.nii.gz")))
    # print("train images:", (os.path.basename(train_images[0])).split('.')[0])

    # Create Data Dictionaries
    train_data = [{"image": img, "report": report_dict[os.path.basename(img).split('.')[0]]} for img in train_images if (os.path.basename(img)).split('.')[0] in report_dict]
    val_data = [{"image": img, "report": report_dict[os.path.basename(img).split('.')[0]]} for img in val_images if (os.path.basename(img)).split('.')[0] in report_dict]
    # print("train data:", train_data)

    # Define Transforms
    # Define transforms
    train_transforms = get_train_transforms()

    valid_transforms = get_valid_transforms()

    # Create Dataset and Dataloader
    train_ds = CacheDataset(data=train_data[0:16], transform=train_transforms, cache_num=4, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_data[0:12], transform=valid_transforms, cache_num=4, cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
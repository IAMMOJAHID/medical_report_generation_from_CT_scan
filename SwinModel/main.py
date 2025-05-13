import torch
import argparse
from dataset import get_dataloaders
from train import train_model
# from model import PaligammaMedicalModel
from SwinModel import FeatureMapTextGenerator


# def parse_agrs():
#     parser = argparse.ArgumentParser()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load models
model = FeatureMapTextGenerator().to(device)

# Load data
train_loader, val_loader = get_dataloaders("../AMOS_MM/imagesTr", "../AMOS_MM/imagesVa", "../AMOS_MM/output.xlsx")
print("Data Loading Done")
# Train the model
train_model(model, train_loader, val_loader, device, epochs=2)

import torch
import argparse
from swin_unetr_model import load_swin_unetr
from bio_bert_model import load_bio_bert
from dataset import get_dataloaders
from train import train_model


# def parse_agrs():
#     parser = argparse.ArgumentParser()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
swin_unetr = load_swin_unetr(device)
tokenizer, bio_bert = load_bio_bert(device)

# Load data
train_loader, val_loader = get_dataloaders("../AMOS_MM/imagesTr", "../AMOS_MM/imagesTr", "../AMOS_MM/output.xlsx")
print("Data Loading Done")
# Train the model
train_model(swin_unetr, bio_bert, train_loader, val_loader, tokenizer, device, epochs=10)

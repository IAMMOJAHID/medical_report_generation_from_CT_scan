### train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from validate import evaluate_model


def train_model(model, train_loader, val_loader, device, epochs=10):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # loss_cal = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        total_loss = 0
        for _, data in enumerate(train_loader):
            images, reports = data["image"].to(device), data["report"] #Average image shape: (350, 350, 400)
            loss = model(image=images, text=reports)
            
            optimizer.zero_grad()          
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
        evaluate_model(model, val_loader, device, epoch)
        # Save the trained models
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss/len(train_loader),
                    }, f'Weights{epoch}.pth')

        print(f'weights Saved for epoch {epoch}')

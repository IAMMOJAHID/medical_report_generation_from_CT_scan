### train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Define a linear projection layer to map Swin-UNETR features to BERT-compatible embeddings
class FeatureToTextAdapter(nn.Module):
    def __init__(self, in_dim, bert_hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_dim, bert_hidden_size)  # Convert image features to text embedding space

    def forward(self, features):
        projected = self.linear(features)  # Map to BERT token space
        integer_values = torch.round(projected).to(torch.long)  # Convert floats to nearest integers
        return integer_values


def evaluate_model(swin_unetr, bio_bert, val_loader, tokenizer, device, epoch):
    swin_unetr.eval()
    bio_bert.eval()
    bleu = Bleu(4)
    rouge = Rouge()
    meteor = Meteor()
    cider = Cider()
    results = []
    
    # Initialize the adapter
    feature_adapter = FeatureToTextAdapter(in_dim=7070976, bert_hidden_size=512).to(device)  # Adjust input size based on swin_unetr output
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, reports = data["image"].to(device), data["report"]
            tokenized_reports = tokenizer(reports, padding="max_length", truncation=True, return_tensors="pt", max_length=512).input_ids.to(device)
            
            feature_maps = swin_unetr.swinViT(images)
            features = torch.cat([fm.flatten(start_dim=1) for fm in feature_maps], dim=1)
            bert_inputs = feature_adapter(features)
            output = bio_bert(bert_inputs).last_hidden_state
            
            generated_texts = [tokenizer.decode(out.argmax(dim=-1).tolist()) for out in output]
            
            for gt, pred in zip(reports, generated_texts):
                bleu_scores, _ = bleu.compute_score({0: [gt]}, {0: [pred]})
                rouge_score = rouge.compute_score({0: [gt]}, {0: [pred]})[0]
                meteor_score = meteor.compute_score({0: [gt]}, {0: [pred]})[0]
                cider_score = cider.compute_score({0: [gt]}, {0: [pred]})[0]
                results.append([gt, pred, *bleu_scores, rouge_score, meteor_score, cider_score])
    
    df = pd.DataFrame(results, columns=["Actual Report", "Generated Report", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE", "METEOR", "CIDEr"])
    df.to_excel(f"validation_results_epoch_{epoch}.xlsx", index=False)
    print(f"Validation results saved for epoch {epoch}")

def train_model(swin_unetr, bio_bert, train_loader, val_loader, tokenizer, device, epochs=10):
    swin_unetr.train()
    bio_bert.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(swin_unetr.parameters()) + list(bio_bert.parameters()), lr=1e-4)
    feature_adapter = FeatureToTextAdapter(in_dim=7070976, bert_hidden_size=512).to(device)  # Adjust input size based on swin_unetr output
    for epoch in range(epochs):
        total_loss = 0
        for _, data in enumerate(train_loader):
            images, reports = data["image"].to(device), data["report"]
            tokenized_reports = tokenizer(reports, padding="max_length", truncation=True, return_tensors="pt", max_length=512).input_ids.to(device)
            print("tokenized report:", tokenized_reports)
            
            optimizer.zero_grad()
            
            feature_maps = swin_unetr.swinViT(images)
            features = torch.cat([fm.flatten(start_dim=1) for fm in feature_maps], dim=1)
            bert_inputs = feature_adapter(features)
            output = bio_bert(bert_inputs).last_hidden_state
            
            loss = criterion(output.view(-1, output.size(-1)), tokenized_reports.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
        evaluate_model(swin_unetr, bio_bert, val_loader, tokenizer, device, epoch)
        # Save the trained models
        torch.save({
            f'swin_unetr{epoch}': swin_unetr.state_dict(),
            f'bio_bert{epoch}': bio_bert.state_dict()
        }, "trained_swin_unetr_nlp.pth")
        print(f'weights Saved for epoch {epoch}')

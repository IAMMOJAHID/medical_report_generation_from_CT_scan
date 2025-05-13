import torch
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def evaluate_model(model, val_loader, device, epoch):
    model.eval()
    bleu = Bleu(4)
    rouge = Rouge()
    meteor = Meteor()
    cider = Cider()
    results = []

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, reports = data["image"].to(device), data["report"]
            generated_texts = [model(image=images)]

            for gt, pred in zip(reports, generated_texts):
                bleu_scores, _ = bleu.compute_score({0: [gt]}, {0: [pred]})
                rouge_score = rouge.compute_score({0: [gt]}, {0: [pred]})[0]
                meteor_score = meteor.compute_score({0: [gt]}, {0: [pred]})[0]
                cider_score = cider.compute_score({0: [gt]}, {0: [pred]})[0]
                results.append([gt, pred, *bleu_scores, rouge_score, meteor_score, cider_score])
    
    df = pd.DataFrame(results, columns=["Actual Report", "Generated Report", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE", "METEOR", "CIDEr"])
    df.to_excel(f"results/validation_results_epoch_{epoch}.xlsx", index=False)
    # print(f"Validation results saved for epoch {epoch}")
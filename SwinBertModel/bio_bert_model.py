from transformers import AutoModel, AutoTokenizer

def load_bio_bert(device):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bio_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    return tokenizer, bio_bert
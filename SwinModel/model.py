import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from einops.layers.torch import Rearrange
from SwinTransformer import SwinTransformer

    
class PaligammaMedicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ReportTokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        self.ReportTokenizer.pad_token = self.ReportTokenizer.eos_token
        self.text_decoder = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(self.device)  # T5
        self.BaseModelOutput = BaseModelOutput
        # self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.vision_encoder = SwinTransformer(
            in_chans = 1,
            embed_dim = 48,
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=[2, 2, 6, 2],
            num_heads = (3, 6, 12, 24)
        ).to(self.device)
        self.projection1 = nn.Linear(768, 512)
        self.projection2 = nn.Linear(100, 150)
        # self.projection2 = nn.Linear(32*32*32, 512)
        # self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=3)
        # self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        # self.channel = Rearrange('b 1 (h1 h2) (w1 w2) (d1 d2) -> b (h2 w2 d2) h1 w1 d1', h2 = 8, w2 = 8, d2=8)
        

    def forward(self, image, text=None):
        image_features = self.vision_encoder(image)[4].to(self.device)
        image_features = image_features.view(image_features.shape[0], image_features.shape[1], -1)
        image_features = self.projection2(image_features)
        image_features = image_features.permute(0, 2, 1)
        # image_features = self.projection1(image_features)
        encoder_outputs = self.BaseModelOutput(last_hidden_state=image_features)
        
        if text != None:
            # tokens = self.scibert_tokenizer(text, return_tensors="pt", padding="max_length", max_length=150, truncation=True)
            # text = self.scibert_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            # text = "summarize: " + text
            tokenizedText = self.ReportTokenizer(text, padding="max_length", max_length=150, truncation=True, return_tensors="pt")
            input_ids, attention_mask = tokenizedText.input_ids.squeeze().to(self.device), tokenizedText.attention_mask.squeeze().to(self.device)
            # Ensure batch dimension is preserved
            if input_ids.dim() == 1:  # If shape is (seq_len,), add batch dimension
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            labels = input_ids.clone().to(self.device)
            decoder_input_ids = self.text_decoder.prepare_decoder_input_ids_from_labels(labels).to(self.device)
            if decoder_input_ids.dim() == 1:
                decoder_input_ids = decoder_input_ids.unsqueeze(0)
            outputs = self.text_decoder(labels=labels, input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)
        else:
            # generated_ids = self.text_decoder.generate(encoder_outputs=encoder_outputs)
            decoder_input_ids = torch.tensor([[self.text_decoder.config.decoder_start_token_id]]).to(self.device)
            generated_ids = self.text_decoder.generate(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                max_length=200,  # Maximum length of the generated text
                min_length=140,   # Minimum length before stopping
                num_beams=7,     # Beam search for better quality
                temperature=0.7,  # Sampling temperature (lower = more deterministic)
                top_k=50,        # Top-k sampling (filters unlikely words)
                top_p=0.95,      # Nucleus sampling (controls randomness)
                repetition_penalty=1.2,  # Penalizes repeated phrases
                early_stopping=True,  # Stops generation when an end token is likely
                no_repeat_ngram_size=5
            )           
            outputs = self.ReportTokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return outputs

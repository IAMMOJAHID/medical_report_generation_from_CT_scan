import torch
import torch.nn as nn
from SwinTransformer import SwinTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F


class FeatureMapTextGenerator(nn.Module):
    def __init__(self, beam_size=5):
        super(FeatureMapTextGenerator, self).__init__()
        self.max_length = [60, 50, 40, 30, 20]
        self.beam_size = beam_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BLIP setup
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        self.feature_map_channels = [48, 96, 192, 384, 768]
        self.feature_projection = nn.ModuleList([
            nn.Conv3d(in_channels, 768, kernel_size=1) for in_channels in self.feature_map_channels
        ])

        self.vision_encoder = SwinTransformer(
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=[2, 2, 6, 2],
            num_heads=(3, 6, 12, 24)
        ).to(self.device)

    def forward(self, image, text=None):
        feature_maps = self.vision_encoder(image)
        combined_features = []

        for i, feature_map in enumerate(reversed(feature_maps)):
            projected = self.feature_projection[len(feature_maps) - 1 - i](feature_map)
            projected = projected.view(projected.size(0), projected.size(1), -1)  # Flatten spatial dims
            projected = projected.permute(0, 2, 1)  # [B, N, D]
            print("Shape of Projected:", projected.shape)

            # Create pseudo-image embedding
            image_embedding = projected.mean(dim=1).view(-1, 1, 24, 32)
            image_embedding = image_embedding.repeat(1, 3, 1, 1)  # [B, 3, 24, 32]
            image_embedding = image_embedding.to(torch.float32)
            image_embedding = (image_embedding - image_embedding.min()) / (image_embedding.max() - image_embedding.min() + 1e-5)
            print("Shape of Image Embedding:", image_embedding.shape)

            # Run BLIP processor & model in generation mode
            inputs = self.processor(images=image_embedding, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs, max_length=self.max_length[i], num_beams=self.beam_size)
            combined_features.append(generated_ids)

        print("len of combined feature:", len(combined_features))
        # Final Output
        if text is not None:
            print("text:", text)
            decoded_texts = [self.processor.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in combined_features]
            decoded_texts = [decoded_texts[0]+" "+decoded_texts[1]+" "+decoded_texts[2]+" "+decoded_texts[3]+" "+decoded_texts[4]]
            if isinstance(decoded_texts, str):
                decoded_texts = [decoded_texts]
            print("decoded_texts:", decoded_texts)
            pseudo_image = image_embedding.detach()
            print("pseudo_image shape:", pseudo_image.shape)

            # Ensure batch sizes match
            if pseudo_image.shape[0] == 1 and len(decoded_texts) > 1:
                # Option 1: Use only the first caption (if single-image batch)
                decoded_texts = [decoded_texts[0]]
                text = [text[0]]  # Ensure target is also batch=1
            else:
                # Option 2: Repeat image to match captions (if multi-caption training)
                pseudo_image = pseudo_image.repeat(len(decoded_texts), 1, 1, 1)
                text = text * len(decoded_texts)

            # Ensure batch sizes match
            assert len(decoded_texts) == pseudo_image.shape[0], f"Batch mismatch: {len(decoded_texts)} captions vs {pseudo_image.shape[0]} images"
            assert len(decoded_texts) == len(text), f"Batch mismatch: {len(decoded_texts)} captions vs {len(text)} targets"
            
            inputs = self.processor(images=pseudo_image, text=decoded_texts, text_target=text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            return outputs.loss

        else:
            # Decode all generated texts
            decoded_texts = [self.processor.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in combined_features]
            # Combine them into one input (you could also join with separators if desired)
            combined_input = " ".join(decoded_texts)
            pseudo_image = image_embedding.detach()
            # Run summarization using BLIP (optional – may not improve result unless fine-tuned for summarization)
            inputs = self.processor(images=pseudo_image, text=combined_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
            summary_ids = self.model.generate(**inputs, max_length=200, num_beams=self.beam_size)

            # Return summarized text
            summarized_text = self.processor.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summarized_text



import numpy as np
import torch
from transformers import AutoTokenizer
from project_inference import ModelForSequenceClassification, average_pool


class SentenceClassifier:

    def load_from_file(self, model_path, tokenizer_path, device):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)           
            self.model = ModelForSequenceClassification().to(device)
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.device = device           
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise 
    
    def predict_tone(self, text: list[str]):
        try:
            self.model.eval()
            tones = []
            with torch.no_grad():
                for i in text:
                    i = i[0].lower() + i[1: ]
                    tones.append(int(np.argmax(self.model.inference(self.tokenizer, i, device=self.device))))
                return tones
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise



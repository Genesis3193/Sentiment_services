import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_utils import ModuleUtilsMixin

MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_SEQ_LENGTH = 512

def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ModelForSequenceClassification(nn.Module, ModuleUtilsMixin):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype="auto",
        )
        config = self.model.config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = average_pool(outputs.last_hidden_state, attention_mask)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def inference(self, tokenizer, text, device):
        inputs = tokenizer(
            [text], max_length=MAX_SEQ_LENGTH,
            padding=False, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = self.forward(**inputs)
            probs = F.softmax(logits, dim=1)
        return probs[0].cpu().tolist()


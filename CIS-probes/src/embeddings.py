import torch
from transformers import GPT2Tokenizer, GPT2Model
from typing import List, Dict

class GPT2CountryEmbeddings:
    def __init__(self, model_name: str = 'gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, country: str) -> torch.Tensor:
        inputs = self.tokenizer(country, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the first token's embedding (country is single-token)
        return outputs.last_hidden_state[0, 0, :].cpu()

    def get_embeddings(self, countries: List[str]) -> Dict[str, torch.Tensor]:
        return {country: self.get_embedding(country) for country in countries} 
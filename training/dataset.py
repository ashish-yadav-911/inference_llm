import json
from torch.utils.data import Dataset


class JsonDataset(Dataset):

    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        text = self.samples[idx]["text"]

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze(),
            "labels": tokens.input_ids.squeeze()
        }
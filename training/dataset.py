"""
Dataset — fixed vs original:
  - No padding in __getitem__: tokenize without padding, let the collator handle it
    (padding="max_length" caused every sample to be 2048 tokens of mostly padding)
  - Collator applies dynamic padding to the actual batch max length
  - Labels are padded with -100 so CrossEntropyLoss ignores padding positions
"""

import json
from torch.utils.data import Dataset


class JsonDataset(Dataset):

    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]

        # Tokenize WITHOUT padding — collator handles per-batch dynamic padding
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = tokens.input_ids.squeeze(0)       # [seq_len]
        attention_mask = tokens.attention_mask.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Labels = input_ids for causal LM; collator will mask padding with -100
            "labels": input_ids.clone(),
        }
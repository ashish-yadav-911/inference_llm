"""
Collator — fixed vs original:
  - Dynamic padding: pads each batch to its own max length, not global max_seq_len
    This avoids wasting compute on padding tokens in short sequences
  - Labels padded with -100 so CrossEntropyLoss ignores padding positions
    (original set labels == pad_token_id which trained on padding = wrong signal)
"""

import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForCausalLM:

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids_list      = [x["input_ids"] for x in batch]
        attention_mask_list = [x["attention_mask"] for x in batch]
        labels_list         = [x["labels"] for x in batch]

        # Pad input_ids and attention_mask with pad_token_id / 0
        input_ids      = pad_sequence(input_ids_list,      batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # Pad labels with -100 so loss ignores padding positions
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
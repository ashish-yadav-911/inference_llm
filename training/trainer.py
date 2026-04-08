import torch


class Trainer:

    def __init__(self, model, optimizer, loader, device):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.device = device

    def train_epoch(self):

        self.model.train()

        for batch in self.loader:

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print("loss:", loss.item())
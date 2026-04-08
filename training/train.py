import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import JsonDataset
from trainer import Trainer
from ddp import setup_ddp, cleanup_ddp


def main():

    local_rank = setup_ddp()

    config = yaml.safe_load(open("training/configs/qwen_train.yaml"))

    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank])

    dataset = JsonDataset(
        config["data"]["dataset_path"],
        tokenizer,
        config["training"]["max_seq_len"]
    )

    sampler = DistributedSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    trainer = Trainer(model, optimizer, loader, local_rank)

    for epoch in range(config["training"]["epochs"]):
        trainer.train_epoch()

    if local_rank == 0:
        model.module.save_pretrained(config["output"]["save_dir"])
        tokenizer.save_pretrained(config["output"]["save_dir"])

    cleanup_ddp()


if __name__ == "__main__":
    main()
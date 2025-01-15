from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from typing_extensions import Literal
import wandb

from data.samplers import InfiniteSampler
from utils import LinearWarmUp, parse_yaml
from train import get_dataset, get_audio_codec, get_tokenizer, get_llm, get_optimizer_and_scheduler, get_audio_and_caption, get_loss_mask, ce_loss, validate



def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    batch_size = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    test_every_n_steps = configs["train"]["test_every_n_steps"]
    save_every_n_steps = configs["train"]["save_every_n_steps"]
    training_steps = configs["train"]["training_steps"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")
    test_dataset = get_dataset(configs, split="test")

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    
    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True
    )

    # Audio encoder: Used to convert audio into discrete codes
    codec = get_audio_codec(configs)

    # Tokenizer: Used to convert text or audio codes into IDs and vice versa
    tokenizer = get_tokenizer(configs, codec)
    vocab_size = len(tokenizer)

    # LLM decoder
    llm = get_llm(configs, vocab_size)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=llm.parameters())

    # Prepare for multiprocessing
    accelerator = Accelerator()

    codec, llm, optimizer, train_dataloader = accelerator.prepare(
        codec, llm, optimizer, train_dataloader)

    if wandb_log and accelerator.is_main_process:
        wandb.init(project="music_llm", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Prepare audio and captions
        audio, captions = get_audio_and_caption(data)
        # audio: (b, c, t_audio), captions: (b, t_text)

        # Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(audio.device)  # shape: (b, t_text)

        audio_codes = codec.module.encode(audio=audio)  # shape: (b, t_code, q)

        # Tokenize audio codes to IDs
        audio_ids = tokenizer.audio_codes_to_ids(audio_codes)  # shape: (b, t_text)

        # Concatenate text and audio IDs along time axis
        ids = torch.cat((caption_ids, audio_ids), dim=1)  # shape: (b, t)
        input_ids = ids[:, 0 : -1]  # (b, t)
        target_ids = ids[:, 1 :]  # (b, t)

        # Loss mask
        loss_mask = get_loss_mask(caption_ids, audio_ids)  # shape: (b, t)

        # Forward
        llm.train()
        logits = llm(ids=input_ids)  # shape: (b, t, v)

        # Loss
        loss = ce_loss(
            output=logits, 
            target=target_ids, 
            mask=loss_mask, 
            ignore_index=tokenizer.pad_token_id
        )
        
        # Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Evaluate
        if step % test_every_n_steps == 0 and accelerator.is_main_process:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                tokenizer=tokenizer, 
                codec=accelerator.unwrap_model(codec), 
                llm=accelerator.unwrap_model(llm)
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                tokenizer=tokenizer, 
                codec=accelerator.unwrap_model(codec), 
                llm=accelerator.unwrap_model(llm)
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
        
        # Save model
        if step % save_every_n_steps == 0 and accelerator.is_main_process:
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(accelerator.unwrap_model(llm).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
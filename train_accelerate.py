from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from music_llm.losses import ce_loss
from music_llm.samplers.infinite_sampler import InfiniteSampler
from music_llm.utils import parse_yaml
from train import (get_audio_and_caption, get_audio_encoder, get_dataset,
                   get_loss_mask, get_model, get_optimizer_and_scheduler,
                   get_tokenizer, validate)


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
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

    # Audio encoder
    audio_encoder = get_audio_encoder(configs)

    # Tokenizer
    tokenizer = get_tokenizer(configs, audio_encoder)
    vocab_size = len(tokenizer)

    # Model
    model = get_model(
        configs=configs, 
        vocab_size=len(tokenizer), 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    )

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=model.parameters())

    # Prepare for multiprocessing
    accelerator = Accelerator()

    audio_encoder, model, optimizer, train_dataloader = accelerator.prepare(
        audio_encoder, model, optimizer, train_dataloader)

    if wandb_log and accelerator.is_main_process:
        wandb.init(project="music_llm", name=config_name)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Prepare audio and captions
        audio, captions = get_audio_and_caption(data)  # audio: (b, c, l_audio), captions: (b, l_text)

        # 1.2 Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(audio.device)  # (b, l_text)

        # 1.3 Encode audio into discrete codes
        audio_codes = audio_encoder.module.encode(audio=audio)  # (b, l_code, q)

        # 1.4 Tokenize audio codes to IDs
        audio_ids = tokenizer.audio_codes_to_ids(audio_codes)  # (b, l_text)

        # 1.5 Concatenate text and audio IDs
        ids = torch.cat((caption_ids, audio_ids), dim=1)  # shape: (b, l)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        logits = model(ids=ids)  # shape: (b, l, v)

        # 2.2 Targets
        out = logits[:, 0 : -1, :]
        target_ids = ids[:, 1 :]
        mask = get_loss_mask(caption_ids, audio_ids[:, 0: -1])  # (b, l)

        # 2.3 Loss
        loss = ce_loss(
            output=out, 
            target=target_ids, 
            mask=mask, 
            ignore_index=tokenizer.pad_token_id
        )
        
        # 2.4 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % test_every_n_steps == 0 and accelerator.is_main_process:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                tokenizer=tokenizer, 
                audio_encoder=accelerator.unwrap_model(audio_encoder), 
                model=accelerator.unwrap_model(model)
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                tokenizer=tokenizer, 
                audio_encoder=accelerator.unwrap_model(audio_encoder), 
                model=accelerator.unwrap_model(model)
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
        
        # 3.2 Save model
        if step % save_every_n_steps == 0 and accelerator.is_main_process:
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == training_steps:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
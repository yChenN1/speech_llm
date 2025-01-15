from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from audidata.io.crops import RandomCrop, StartCrop
from audidata.transforms import Mono, Normalize, TimeShift
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import Literal

import wandb
from data.samplers import InfiniteSampler
from llm.llama import Llama, LlamaConfig
from utils import LinearWarmUp, parse_yaml


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
    codec.to(device)

    # Tokenizer: Used to convert text or audio codes into IDs and vice versa
    tokenizer = get_tokenizer(configs, codec)
    vocab_size = len(tokenizer)

    # LLM decoder
    llm = get_llm(configs, vocab_size)
    llm.to(device)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=llm.parameters()
    )

    if wandb_log:
        wandb.init(project="music_llm", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Get audio and captions
        audio, captions = get_audio_and_caption(data)
        # audio: (b, c, t_audio), captions: (b, t_text)

        # Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(device)  # shape: (b, t_text)
        
        # Encode audio into discrete codes
        audio = audio.to(device)
        audio_codes = codec.encode(audio=audio)  # shape: (b, t_code, q)

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
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Evaluate
        if step % test_every_n_steps == 0:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                tokenizer=tokenizer, 
                codec=codec, 
                llm=llm
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                tokenizer=tokenizer, 
                codec=codec, 
                llm=llm
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))

        # Save model
        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(llm.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break

        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]

    datasets = []
    datasets_split = "{}_datasets".format(split)

    for name in configs[datasets_split].keys():
    
        if name == "FreeSpokenDigit":

            from audidata.datasets import FreeSpokenDigit

            dataset = FreeSpokenDigit(
                root=configs[datasets_split][name]["root"],
                split=split,
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=[Mono(), Normalize(), TimeShift(sr=sr, shift=(0., 0.5))],
            )
            datasets.append(dataset)

        elif name == "LJSpeech":

            from audidata.datasets import LJSpeech

            dataset = LJSpeech(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration), 
                transform=[Mono(), Normalize(), TimeShift(sr=sr, shift=(0., 0.5))],
            )
            datasets.append(dataset)

        elif name == "GTZAN":

            from audidata.datasets import GTZAN

            dataset = GTZAN(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                test_fold=configs[datasets_split][name]["test_fold"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=None
            )
            datasets.append(dataset)

        elif name == "Shutterstock":

            from audidata.datasets import Shutterstock

            dataset = Shutterstock(
                root=configs[datasets_split][name]["root"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=None
            )
            datasets.append(dataset)

        else:
            raise ValueError(name)

        # Todo: return multiple datasets
        return dataset


def get_audio_codec(configs: dict) -> nn.Module:
    r"""Load pretrained audio encoder."""

    name = configs["audio_codec"]["name"]
    
    if name == "DAC":
        
        from audio_codec.dac import DAC

        return DAC(
            sr=configs["sample_rate"],
            n_quantizers=configs["audio_codec"]["n_quantizers"]
        )

    elif name == "XCodec2":

        from audio_codec.xcodec import XCodec2
        
        return XCodec2(sr=configs["sample_rate"])

    else:
        raise ValueError(name)


def get_tokenizer(configs: dict, codec: nn.Module) -> object:
    r"""Get tokenizer with merged text and audio tokens."""

    name = configs["tokenizer"]["name"]

    if name == "BertDAC":

        from tokenizer.bert_dac import BertDacTokenizer

        return BertDacTokenizer(codec)

    elif name == "BertXCodec2":

        from tokenizer.bert_xcodec import BertXCodecTokenizer
        
        return BertXCodecTokenizer(codec)

    else:
        raise ValueError(name)


def get_llm(configs: dict, vocab_size: int) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["llm"]["name"]

    if name == "Llama":

        block_size = configs["llm"]["block_size"]
        n_layer = configs["llm"]["n_layer"]
        n_head = configs["llm"]["n_head"]
        n_embd = configs["llm"]["n_embd"]

        config = LlamaConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd
        )
        return Llama(config=config)

    else:
        raise ValueError(name)    


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def get_audio_and_caption(data: dict) -> tuple[torch.Tensor, list[str]]:
    r"""Process data to audio and captions according to different datasets."""

    name = data["dataset_name"]

    if isinstance(name, list):
        name = name[0]
    
    if name in ["GTZAN"]:
        return data["audio"], data["label"]

    elif name in ["FreeSpokenDigit", "LJSpeech", "Shutterstock"]:
        return data["audio"], data["caption"]

    else:
        raise ValueError(name)


def get_loss_mask(
    caption_ids: torch.LongTensor, 
    audio_ids: torch.LongTensor
) -> torch.Tensor:
    r"""Get loss mask to train the autoregression model.

    Args:
        caption_ids: (b, t1)
        audio_ids: (b, t2)

    Returns:
        mask: (b, t1+t2-1)
    """

    device = caption_ids.device
    B, T1 = caption_ids.shape
    T2 = audio_ids.shape[1]

    caption_mask = torch.zeros((B, T1))
    audio_mask = torch.ones((B, T2 - 1))

    mask = torch.cat((caption_mask, audio_mask), dim=1).to(device)

    return mask


def ce_loss(
    output: torch.Tensor, 
    target: torch.LongTensor, 
    mask: torch.Tensor,
    ignore_index: int
) -> float:
    r"""Cross entropy loss.

    b: batch_size
    t: time_steps
    v: vocab_size

    Args:
        output: (b, t, v), logits
        target: (b, t)
        mask: (b, t)
        ignore_index: int

    Outputs:
        loss: torch.float
    """

    B, T, V = output.shape

    loss = F.cross_entropy(
        input=output.flatten(0, 1),  # shape: (b*t, v)
        target=target.flatten(0, 1),  # shape: (b*t,)
        ignore_index=ignore_index,
        reduction="none"
    )  # shape: (b*t,)

    loss = loss * mask.flatten(0, 1)  # shape: (b*t,)
    loss = torch.mean(loss)

    return loss


def validate(
    configs: dict,
    dataset: Dataset,
    tokenizer: object,
    codec: nn.Module, 
    llm: nn.Module, 
    valid_steps=100
) -> float:
    r"""Validate the model on part of data."""

    device = next(codec.parameters()).device
    losses = []

    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
    
        data = dataset[idx]
        
        # Get audio and captions
        audio, caption = get_audio_and_caption(data)
        # audio: (c, t), caption: str
        
        # Batch data
        audio = torch.Tensor(audio[None, :, :]).to(device)  # shape: (b, c, t_audio)
        captions = [caption]  # shape: (b, t_text)
        
        # Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(device)  # shape: (b, t_text)

        # Encode audio into discrete codes
        audio = audio.to(device)
        audio_codes = codec.encode(audio=audio)  # shape: (b, t_code, q)
        
        # Tokenize audio codes to IDs
        audio_ids = tokenizer.audio_codes_to_ids(audio_codes)  # shape: (b, t_text)

        # Concatenate text and audio IDs along time axis
        ids = torch.cat((caption_ids, audio_ids), dim=1)  # shape: (b, t)
        input_ids = ids[:, 0 : -1]  # (b, t)
        target_ids = ids[:, 1 :]  # (b, t)

        # Loss mask
        loss_mask = get_loss_mask(caption_ids, audio_ids)  # shape: (b, t)
        
        # Forward
        with torch.no_grad():
            llm.eval()
            logits = llm(ids=input_ids)  # shape: (b, t, v)

        # Loss
        loss = ce_loss(
            output=logits, 
            target=target_ids, 
            mask=loss_mask, 
            ignore_index=tokenizer.pad_token_id
        )

        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
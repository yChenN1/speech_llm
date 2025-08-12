from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import wandb
from music_llm.losses import ce_loss
from music_llm.samplers.infinite_sampler import InfiniteSampler
from music_llm.utils import LinearWarmUp, parse_yaml


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

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
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Audio encoder
    audio_encoder = get_audio_encoder(configs).to(device)

    # Tokenizer
    tokenizer = get_tokenizer(configs, audio_encoder)

    # Model
    model = get_model(
        configs=configs, 
        vocab_size=len(tokenizer), 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    if wandb_log:
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
        ).to(device)  # (b, l_text)
        
        # 1.3 Encode audio into discrete codes
        audio = audio.to(device)
        audio_codes = audio_encoder.encode(audio=audio)  # (b, l_code, q)

        # 1.4 Tokenize audio codes to IDs
        audio_ids = tokenizer.audio_codes_to_ids(audio_codes)  # (b, l_text)

        # 1.5 Concatenate text and audio IDs
        ids = torch.cat((caption_ids, audio_ids), dim=1)  # (b, l)
        
        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        logits = model(ids)  # shape: (b, l, v)

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
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                tokenizer=tokenizer, 
                audio_encoder=audio_encoder, 
                model=model
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                tokenizer=tokenizer, 
                audio_encoder=audio_encoder, 
                model=model
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {:.2f}".format(train_loss))
            print("Test loss: {:.2f}".format(test_loss))

        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break

        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    ds = f"{split}_datasets"

    for name in configs[ds].keys():

        if name == "LJSpeech":

            from audidata.io.crops import StartCrop
            from audidata.transforms.audio import Mono, Normalize, TimeShift

            from music_llm.datasets.ljspeech import LJSpeech

            dataset = LJSpeech(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration), 
                transform=[Mono(), Normalize(), TimeShift(sr=sr, shift=(0., 0.5))],
            )
            return dataset
            
        elif name == "GTZAN":

            from audidata.io.crops import RandomCrop
            from audidata.transforms.audio import Mono

            from music_llm.datasets.gtzan import GTZAN

            dataset = GTZAN(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                test_fold=configs[ds][name]["test_fold"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=None
            )
            return dataset

        elif name == "Shutterstock":

            from audidata.datasets import Shutterstock
            from audidata.io.crops import RandomCrop
            from audidata.transforms.audio import Mono

            dataset = Shutterstock(
                root=configs[ds][name]["root"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=None
            )
            return dataset

        elif name == 'InstructSpeech':
            from audidata.transforms.audio import Mono
            from music_llm.datasets.Instructspeech import InstructSpeech
            dataset = InstructSpeech(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                transform=Mono(),
                target_transform=None
            )

        else:
            raise ValueError(name)

        # Todo: return multiple datasets
        return dataset


def get_audio_encoder(configs: dict) -> nn.Module:
    r"""Load pretrained audio encoder."""

    name = configs["audio_encoder"]["name"]
    
    if name == "DAC":        
        from music_llm.audio_encoders.dac import DAC
        return DAC(
            sr=configs["sample_rate"],
            n_quantizers=configs["audio_encoder"]["n_quantizers"]
        )

    elif name == "XCodec2":
        from music_llm.audio_encoders.xcodec import XCodec2
        return XCodec2(sr=configs["sample_rate"])

    else:
        raise ValueError(name)


def get_tokenizer(configs: dict, audio_encoder: nn.Module) -> object:
    r"""Get tokenizer with merged text and audio tokens."""

    name = configs["tokenizer"]["name"]

    if name == "BertDAC":
        from music_llm.tokenizers.bert_dac import BertDacTokenizer
        return BertDacTokenizer(
            codecbook_size=audio_encoder.codec.codebook_size, 
            n_quantizers=audio_encoder.n_quantizers
        )
        
    elif name == "BertXCodec2":
        from music_llm.tokenizers.bert_xcodec import BertXCodecTokenizer
        return BertXCodecTokenizer(audio_encoder.vocab_size)

    elif name == "BertXCodec2ATA":
        from music_llm.tokenizers.bert_xcodec_ata import BertXCodec2ATATokenizer
        return BertXCodec2ATATokenizer(audio_encoder.vocab_size)

    else:
        raise ValueError(name)


def get_model(configs: dict, vocab_size: int, ckpt_path: str) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["model"]["name"]

    if name == "Llama":
        from music_llm.models.llama import Llama, LlamaConfig
        config = LlamaConfig(
            block_size=configs["model"]["block_size"],
            vocab_size=vocab_size,
            n_layer=configs["model"]["n_layer"],
            n_head=configs["model"]["n_head"],
            n_embd=configs["model"]["n_embd"],
        )
        model = Llama(config)
    else:
        raise ValueError(name)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


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

    elif name in ['InstructSpeech']:
        return data["src_audio"], data['trg_audio'], data["caption"], data['src_audio_valid_length'], data['trg_audio_valid_length']
    else:
        raise ValueError(name)


def get_loss_mask(
    caption_ids: LongTensor, 
    audio_ids: LongTensor
) -> torch.Tensor:
    r"""Get loss mask to train the autoregression model.

    Args:
        caption_ids: (b, l1)
        audio_ids: (b, l2)

    Returns:
        mask: (b, l1+l2)
    """

    mask = torch.cat((torch.zeros_like(caption_ids), torch.ones_like(audio_ids)), dim=1)
    return mask


def validate(
    configs: dict,
    dataset: Dataset,
    tokenizer: object,
    audio_encoder: nn.Module, 
    model: nn.Module, 
    valid_steps=100
) -> float:
    r"""Validate the model on part of data."""

    device = next(audio_encoder.parameters()).device
    losses = []

    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
    
        # ------ 1. Data preparation ------
        # 1.0 Get Data
        data = dataset[idx]
        data = default_collate([data])

        # 1.1 Prepare audio and captions
        src_audio, trg_audio, captions, src_audio_valid_length, trg_audio_valid_length = get_audio_and_caption(data)  # audio: (b, c, l_audio), captions: (b, l_text)

        # 1.2 Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(device)  # (b, l_text)
        
        # 1.3 Encode audio into discrete codes
        src_audio_codes = audio_encoder.encode(audio=src_audio)  # (b, l_code, q)
        trg_audio_codes = audio_encoder.encode(audio=trg_audio)  # (b, l_code, q)
        
        # 1.4 Tokenize audio codes to IDs
        src_audio_ids = tokenizer.audio_codes_to_ids(src_audio_codes, type="src")  # (b, l_text)
        trg_audio_ids = tokenizer.audio_codes_to_ids(trg_audio_codes, type="trg")  # (b, l_text)

        # 1.5 Concatenate text and audio IDs
        ids = torch.cat((caption_ids, src_audio_ids, trg_audio_ids), dim=1)  # shape: (b, l)

        # 1.6 Targets
        target_ids = ids[:, 1 :]
        mask = get_loss_mask(torch.cat((caption_ids, src_audio_ids), dim=1), trg_audio_ids[:, 0: -1])  # (b, l)

        # ------ 2. Evaluation ------
        # 2.1 Forward
        with torch.no_grad():
            model.eval()
            logits = model(ids)  # shape: (b, l, v)

        out = logits[:, 0 : -1, :]

        # 2.2 Loss
        loss = ce_loss(
            output=out, 
            target=target_ids, 
            mask=mask, 
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
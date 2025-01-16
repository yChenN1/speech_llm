from __future__ import annotations

import argparse
from pathlib import Path

import torch
import soundfile

from train import get_audio_codec, get_tokenizer, get_llm
from utils import parse_yaml


def sample(args):

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path

    # Configs
    configs = parse_yaml(config_path)

    start_char = "\n"
    # num_samples = 5  # Number of samples to draw
    num_samples = 2  # Number of samples to draw
    # max_new_ids = 2048  # Number of tokens generated in each sample
    # max_new_ids = 1400
    max_new_ids = configs["llm"]["block_size"] - configs["max_caption_len"]
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability
    
    device = "cuda"
    sr = configs["sample_rate"]

    # Audio encoder: Used to convert to convert audio into discrete codes
    codec = get_audio_codec(configs)
    codec.to(device)

    # Tokenizer: Used to convert text/audio codes into IDs and vice versa
    tokenizer = get_tokenizer(configs, codec)
    vocab_size = len(tokenizer)

    # LLM decoder
    llm = get_llm(configs, vocab_size)
    llm.load_state_dict(torch.load(ckpt_path))
    llm.to(device)

    # captions = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"] 
    # captions = ['metal']
    # captions = ["Fun and driving, featuring flute, alto and bari saxophones, upright bass, and drums that create a positive, upbeat mood"]
    # captions = ["Violin playing and bass, fast drums that are happy"]
    # captions = ["A happy dog"]
    # captions = ["Giltspur Street, and the Poultry, or about four hundred and seventy-six in all."] 
    # captions = ["The proposal made was to purchase some fifty thousand square feet between Newgate, Warwick Lane, and the Sessions House"]
    # captions = ["All the latest news, opinions and analysis on Hong Kong, China, Asia and around the world."]
    captions = ["A happy dog ran through the park, wagging its tail excitedly, greeting everyone with joy and boundless energy."]

    # captions = []
    # for i in range(10):
    #     caption = "speaker: {}, label: {}".format("nicolas", i)
    #     captions.append(caption)

    captions_list = [[caption] for caption in captions]

    for captions in captions_list:

        # Tokenize captions to IDs
        caption_ids = tokenizer.captions_to_ids(
            captions=captions, 
            fix_length=configs["max_caption_len"]
        ).to(device)
        # shape: (b, t_text)


        B = len(captions)
        audio_ids = torch.ones(size=(B, 1), dtype=torch.long, device=device) * tokenizer.boa_token_id

        input_ids = torch.cat((caption_ids, audio_ids), dim=1)

        # Sample    
        for n in range(num_samples):

            with torch.no_grad():
                llm.eval()
                ids = llm.generate(
                    ids=input_ids, 
                    max_new_ids=max_new_ids, 
                    temperature=temperature, 
                    top_k=top_k
                )
                # shape: (b, t)

            audio_codes = tokenizer.ids_to_audio_codes(ids)

            audio = codec.decode(audio_codes)
            audio = audio.cpu().numpy()[0, 0]

            # print(n)
            for b in range(B):
                results_dir = Path("results", Path(config_path).stem)
                results_dir.mkdir(parents=True, exist_ok=True)
                audios_path = Path(results_dir, "{}_sample_{}.wav".format(captions[0], n))
                soundfile.write(file=audios_path, data=audio, samplerate=sr)
                print("Write out to {}".format(audios_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    sample(args)
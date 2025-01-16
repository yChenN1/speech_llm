from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

from utils import pad_or_truncate


class BertDacTokenizer:
    r"""Extend text tokenizer with discrete audio codec vocabularies.
    """
    
    def __init__(self, audio_codec: nn.Module) -> None:

        super().__init__()

        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Audio encoder attributes
        self.codebook_size = audio_codec.codec.codebook_size  # E.g., 1024
        self.n_quantizers = audio_codec.n_quantizers  # E.g., 2

        # Begin of audio (boa) and end of audio (eoa) tokens
        new_vocabs = ["<boa>", "<eoa>"]

        # Audio codec vocabs
        for q in range(self.n_quantizers):
            for i in range(self.codebook_size):
                new_vocabs.append("dac_l{}_{}".format(q, i))

        # Merge text tokens and audio tokens
        print("Original vocab size: {}".format(len(self.tok)))
        print("Audio vocab size: {}".format(len(new_vocabs)))
        self.tok.add_tokens(new_vocabs)
        print("Extended vocab size: {}".format(len(self.tok)))

    def captions_to_ids(
        self,
        captions: list[str], 
        fix_length: int
    ) -> torch.LongTensor:
        r"""Convert captions to IDs. 

        E.g., ["Hello world", "rock"]
           -> [[101, 8667, 1362, 102, 0, 0], [101, 2067, 102, 0, 0, 0]]

        Args:
            captions: list[str]
            fix_length: int

        Returns:
            batch_ids: (b, t)
        """

        batch_ids = []

        for caption in captions:
        
            # Convert texts to tokens
            tokens = self.tok.tokenize(caption)

            # Convert tokens to IDs. Reserve 2 IDs for special IDs
            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]

            # Append special IDs
            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            # Pad
            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            batch_ids.append(ids)

        return torch.LongTensor(batch_ids)

    def audio_codes_to_ids(self, codes: torch.LongTensor) -> torch.LongTensor:
        r"""Convert audio codes to tokens, then to IDs.

        E.g.,
            audio_codes: [[[568, 568, 568], [778, 778, 804]]] 

         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]

         -> IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]

        Args:
            codes: (b, q, t)

        Outputs:
            batch_ids: (b, t*q)
        """

        device = codes.device
        B, Q, T = codes.shape

        codes = codes.cpu().numpy()
        batch_ids = np.zeros_like(codes, dtype="int64")

        for b in range(B):
            for q in range(Q):
                for t in range(T):
                    token = "dac_l{}_{}".format(q, codes[b, q, t])
                    batch_ids[b, q, t] = self.tok.convert_tokens_to_ids(token)

        batch_ids = rearrange(batch_ids, 'b q t -> b (t q)')

        # Special tokens
        boa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<boa>")
        eoa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<eoa>")

        batch_ids = np.concatenate((boa_ids, batch_ids, eoa_ids), axis=-1)  # shape: (b, t)
        batch_ids = torch.LongTensor(batch_ids).to(device)

        return batch_ids

    def ids_to_audio_codes(self, ids: torch.LongTensor) -> torch.LongTensor:
        r"""Convert IDs to aduio tokens, then to audio codes.

        E.g.,
            IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]

         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]

         -> audio_codes: [[[568, 568, 568], [778, 778, 804]]]

        Args:
            codes: (b, t*q)

        Outputs:
            batch_ids: (b, q, t)
        """

        device = ids.device
        B, T = ids.shape

        ids = ids.cpu().numpy()
        batch_codes = []

        for b in range(B):
            
            tokens = self.tok.convert_ids_to_tokens(ids[b])
            codes = []

            for t in range(T):
                
                token = tokens[t]
                match = re.match(r'dac_l(\d+)_(\d+)', token)

                if match:

                    q = int(match.groups()[0])
                    id = int(match.groups()[1])

                    if q == 0:
                        buffer = []

                    buffer.append(id)

                    if q == self.n_quantizers - 1:
                        if len(buffer) == self.n_quantizers:
                            codes.append(buffer)
                            buffer = []

            codes = torch.LongTensor(codes)
            codes = rearrange(codes, 't q -> q t')  # shape: (q, t)
            batch_codes.append(codes)

        batch_codes = torch.stack(batch_codes, dim=0).to(device)  # shape: (b, q, t)

        return batch_codes


    def __len__(self):
        return len(self.tok)

    @property
    def pad_token_id(self):
        return self.tok.pad_token_id

    @property
    def boa_token_id(self):
        return self.tok.convert_tokens_to_ids("<boa>")

    @property
    def eoa_token_id(self):
        return self.tok.convert_tokens_to_ids("<eoa>")
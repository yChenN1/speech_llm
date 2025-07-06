from __future__ import annotations

import re

import numpy as np
import torch
from einops import rearrange
from torch import LongTensor
from transformers import AutoTokenizer

from music_llm.utils import pad_or_truncate


class BertDacTokenizer:
    r"""Extend text tokenizer with discrete audio codec vocabularies.
    """
    
    def __init__(
        self, 
        codecbook_size: int,  # E.g., 1024
        n_quantizers: int  # E.g., 2
    ) -> None:

        super().__init__()

        self.codebook_size = codecbook_size
        self.n_quantizers = n_quantizers

        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        new_vocabs = ["<boa>", "<eoa>"]  # <boa>: begin of audio; <eoa>: end of audio

        # Audio codec vocabs
        for q in range(self.n_quantizers):
            for i in range(self.codebook_size):
                new_vocabs.append("dac_q{}_{}".format(q, i))

        # Merge text tokens and audio tokens
        print("Original vocab size: {}".format(len(self.tok)))
        print("Audio vocab size: {}".format(len(new_vocabs)))
        self.tok.add_tokens(new_vocabs)
        print("Extended vocab size: {}".format(len(self.tok)))

    def captions_to_ids(
        self,
        captions: list[str], 
        fix_length: int
    ) -> LongTensor:
        r"""Convert captions to IDs. 

        E.g., ["Hello world", "rock"]
           -> [[101, 8667, 1362, 102, 0, 0], [101, 2067, 102, 0, 0, 0]]

        Args:
            captions: list[str]
            fix_length: int

        Returns:
            batch_ids: (b, l)
        """

        batch_ids = []

        for caption in captions:
        
            # Convert texts to tokens
            tokens = self.tok.tokenize(caption)

            # Convert tokens into IDs, reserving space for two special IDs
            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]
            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            # Pad
            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            batch_ids.append(ids)

        return LongTensor(batch_ids)

    
    def audio_codes_to_ids(self, codes: LongTensor) -> LongTensor:
        r"""Convert audio codes to tokens, then to IDs.

        E.g.,
            audio_codes: [[[568, 568, 568], [778, 778, 804]]] 
         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]
         -> IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]

        Args:
            codes: (b, t, q)

        Outputs:
            batch_ids: (b, t*q)
        """

        device = codes.device
        B, T, Q = codes.shape

        codes = codes.cpu().numpy()
        batch_ids = np.zeros_like(codes, dtype="int64")

        for b in range(B):
            for t in range(T):
                for q in range(Q):
                    token = "dac_q{}_{}".format(q, codes[b, t, q])
                    batch_ids[b, t, q] = self.tok.convert_tokens_to_ids(token)

        batch_ids = rearrange(batch_ids, 'b t q -> b (t q)')

        # Special tokens
        boa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<boa>")
        eoa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<eoa>")

        batch_ids = np.concatenate((boa_ids, batch_ids, eoa_ids), axis=-1)  # (b, t)
        batch_ids = LongTensor(batch_ids).to(device)

        return batch_ids

    def ids_to_audio_codes(self, ids: LongTensor) -> LongTensor:
        r"""Convert IDs to aduio tokens, then to audio codes.

        E.g.,
            IDs: [[30522, 31092, 32326, 31092, 32326, 31092, 32352, 30523]]
         -> tokens: [["<boa>", "dac_l0_568", "dac_l1_778", "dac_l0_568", 
                      "dac_l1_778", "dac_l0_568", "dac_l1_804", "<eoa>"]]
         -> audio_codes: [[[568, 568, 568], [778, 778, 804]]]

        Args:
            codes: (b, t*q)

        Outputs:
            batch_ids: (b, t, q)
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
                match = re.match(r'dac_q(\d+)_(\d+)', token)

                if not match:
                    continue

                q = int(match.groups()[0])
                id = int(match.groups()[1])

                if q == 0:
                    buffer = []

                buffer.append(id)

                if q == self.n_quantizers - 1:
                    if len(buffer) == self.n_quantizers:
                        codes.append(buffer)
                        buffer = []

            codes = LongTensor(codes)
            batch_codes.append(codes)

        batch_codes = torch.stack(batch_codes, dim=0).to(device)  # shape: (b, t, q)

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
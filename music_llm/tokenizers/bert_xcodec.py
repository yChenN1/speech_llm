import re

import numpy as np
from einops import rearrange
from torch import LongTensor
from transformers import AutoTokenizer

from music_llm.utils import pad_or_truncate


class BertXCodecTokenizer:
    def __init__(self, vocab_size: int) -> None:

        super().__init__()

        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        new_vocabs = ["<boa>", "<eoa>"]  # <boa>: begin of audio; <eoa>: end of audio

        # XCodec tokens
        n_quantizers = 1
        for q in range(n_quantizers):
            for i in range(vocab_size):
                new_vocabs.append("xcodec_q{}_{}".format(q, i))

        # Merge text tokens and audio tokens
        print("Original vocab size: {}".format(len(self.tok)))
        print("Audio vocab size: {}".format(len(new_vocabs)))
        self.tok.add_tokens(new_vocabs)
        print("Extended vocab size: {}".format(len(self.tok)))

    def captions_to_ids(
        self,
        captions: list[str], 
        fix_length: None | int
    ) -> LongTensor:
        r"""Convert texts to IDs. 
        E.g., ["classical", "reggae"] -> [[101, 4556, 102], [101, 15662, 102]]
        """

        batch_ids = []

        for caption in captions:
            
            # Convert texts to tokens
            tokens = self.tok.tokenize(caption)

            # Convert tokens into IDs, reserving space for two special IDs
            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]
            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            batch_ids.append(ids)

        return LongTensor(batch_ids)

    def audio_codes_to_ids(self, codes: LongTensor) -> LongTensor:
        r"""Convert audio codes to tokens, then to IDs.
        E.g. [[[471, 330, ...]]] -> [[30522, 30995, 31878, ...]]

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
                    token = "xcodec_q{}_{}".format(q, codes[b, t, q])
                    batch_ids[b, t, q] = self.tok.convert_tokens_to_ids(token)

        batch_ids = rearrange(batch_ids, 'b t q -> b (t q)')

        # Special tokens
        boa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<boa>")
        eoa_ids = np.ones((B, 1), dtype="int64") * self.tok.convert_tokens_to_ids("<eoa>")

        batch_ids = np.concatenate((boa_ids, batch_ids, eoa_ids), axis=-1)  # (b, t)
        batch_ids = LongTensor(batch_ids).to(device)
        
        return batch_ids

    def ids_to_audio_codes(self, ids: LongTensor) -> LongTensor:

        device = ids.device
        B, T = ids.shape

        ids = ids.cpu().numpy()
        batch_codes = []

        for b in range(B):
            
            tokens = self.tok.convert_ids_to_tokens(ids[b])
            codes = []

            for t in range(T):
                
                token = tokens[t]
                match = re.match(r'xcodec_(\d+)', token)

                if not match:
                    continue

                id = int(match.groups()[0])
                codes.append(id)

            batch_codes.append(codes)

        batch_codes = LongTensor(batch_codes).to(device)  # shape: (b, t, q)

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
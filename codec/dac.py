from __future__ import annotations

import dac
import torch
import torch.nn as nn


class DAC(nn.Module):
    def __init__(self, sr: float, n_quantizers: int) -> None:
        super().__init__()

        assert sr == 44100

        model_path = dac.utils.download(model_type="44khz")
        self.codec = dac.DAC.load(model_path)
        self.n_quantizers = n_quantizers

    def encode(
        self, 
        audio: torch.Tensor, 
    ) -> torch.LongTensor:
        r"""

        b: batch_size
        c: audio_channels
        q: quantizers_num

        Args:
            audio: (b, c, t)

        Returns:
            codes: (b, q, t)
        """

        with torch.no_grad():
            self.codec.eval()
            _, codes, _, _, _ = self.codec.encode(
                audio_data=audio, 
                n_quantizers=self.n_quantizers
            )
            # codes: (b, q, t), integer, codebook indices

        if self.n_quantizers:
            codes = codes[:, 0 : self.n_quantizers, :]
            # shape: (b, q, t)

        return codes

    def decode(
        self, 
        codes: torch.LongTensor, 
    ) -> torch.Tensor:
        r"""

        b: batch_size
        c: audio_channels
        q: quantizers_num

        Args:
            codes: (b, q, t)
            model: nn.Module
            n_quantizers: int

        Returns:
            audio: (b, c, t)
        """

        with torch.no_grad():
            self.codec.eval()
            z, _, _ = self.codec.quantizer.from_codes(codes)
            audio = self.codec.decode(z)

        return audio
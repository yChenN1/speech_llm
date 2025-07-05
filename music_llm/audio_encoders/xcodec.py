import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor
from xcodec2.modeling_xcodec2 import XCodec2Model


class XCodec2(nn.Module):
    def __init__(self, sr: int) -> None:
        super().__init__()

        assert sr == 16000

        model_path = "HKUST-Audio/xcodec2"
        self.codec = XCodec2Model.from_pretrained(model_path)
        self.vocab_size = 65536

    def encode(
        self, 
        audio: Tensor, 
    ) -> LongTensor:
        r"""

        b: batch_size
        c: audio_channels
        l: audio_samples
        t: time_steps
        q: quantizers_num

        Args:
            audio: (b, c, l)

        Returns:
            codes: (b, t, q)
        """

        B, C = audio.shape[0 : 2]
        assert C == 1, "XCodec only support mono audio."

        with torch.no_grad():
            
            codes = []

            for b in range(B):
                self.codec.eval()
                code = self.codec.encode_code(input_waveform=audio[b : b + 1, 0, :])  # (1, q, t)
                codes.append(code)

            codes = torch.cat(codes, dim=0)  # (b, q, t)
            codes = rearrange(codes, 'b q t -> b t q')

        return codes

    def decode(
        self, 
        codes: LongTensor, 
    ) -> Tensor:
        r"""

        b: batch_size
        c: audio_channels
        l: audio_samples
        t: time_steps

        Args:
            codes: (b, t, 1)
            model: nn.Module
            n_quantizers: int

        Returns:
            audio: (b, c, l)
        """

        B = codes.shape[0]

        with torch.no_grad():

            audios = []

            self.codec.eval()
            for b in range(B):
                audio = self.codec.decode_code(codes[b : b + 1, None, :])  # (1, 1, t)
                audios.append(audio)

            audios = torch.cat(audios, dim=0)  # (b, c, l)
            
        return audios
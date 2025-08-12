from __future__ import annotations

import os
import json
import random
import math
from typing import Literal, List, Dict, Any

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pathlib import Path
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Literal
import librosa
import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import get_worker_info


TOTAL_STRIDE = 320  # 之前你验证的 encoder 总 stride（每 320 个采样点 -> 1 帧）

class InstructSpeech(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"] = "train",
        sr: int = 22050,
        transform = Mono(),
        target_transform = None,
        max_duration: float = 10.0,
        base_path: str = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_train',

        # 一次性扫描 & 缓存
        scan_once: bool = True,
        blacklist_file: str = "long_audio_blacklist.txt",
        lengths_cache: str = "src_lengths.json",

        # 只检查 source（更快）
        check_target: bool = False,
    ) -> None:
        """
        root: 指向 CSV，至少包含 'audio_path', 'vc_path', 'trg_instruct' 列
        """
        self.root = root
        self.split = split
        self.sr = sr
        self.transform = transform
        self.target_transform = target_transform
        self.max_duration = max_duration
        self.base_path = base_path

        self.scan_once = scan_once
        self.blacklist_file = f"{os.path.splitext(self.root)[0]}.{self.split}.blacklist.txt"
        self.lengths_cache = f"{os.path.splitext(self.root)[0]}.{self.split}.src_lengths.json"
        self.check_target = check_target

        self.data: List[Dict[str, Any]] = pd.read_csv(self.root).to_dict(orient="records")

        # ---- 读取黑名单（持久化） ----
        if os.path.exists(self.blacklist_file):
            with open(self.blacklist_file, "r") as f:
                self.blacklist = set(line.strip() for line in f if line.strip())
        else:
            self.blacklist = set()

        # ---- 读取/构建时长缓存（只存 src） ----
        self.src_lengths: List[int] = []
        need_scan = self.scan_once and (not os.path.exists(self.blacklist_file) or not os.path.exists(self.lengths_cache))

        if need_scan:
            print(f"[ScanOnce] 扫描 {len(self.data)} 条数据（仅检查 source={not self.check_target}）...")
            tmp_blacklist = set()
            tmp_lengths = []

            for rec in tqdm(self.data, desc="Scanning", unit="sample"):
                src_audio_path = os.path.join(self.base_path, rec['audio_path'])
                trg_audio_path = rec['vc_path']
                uid = f"{src_audio_path}||{trg_audio_path}"

                # 获取 src 时长
                d_src = self._safe_duration(src_audio_path)
                tmp_lengths.append(int(round(d_src * self.sr)))

                too_long = (d_src > self.max_duration)
                if self.check_target:
                    d_trg = self._safe_duration(trg_audio_path)
                    too_long = too_long or (d_trg > self.max_duration)

                if too_long:
                    tmp_blacklist.add(uid)

            # 写黑名单
            self.blacklist |= tmp_blacklist
            with open(self.blacklist_file, "w") as f:
                for uid in sorted(self.blacklist):
                    f.write(uid + "\n")
            print(f"[ScanOnce] 黑名单生成完成，共 {len(self.blacklist)} 条。")

            # 写长度缓存（采样点数）
            with open(self.lengths_cache, "w") as f:
                json.dump(tmp_lengths, f)
            self.src_lengths = tmp_lengths
            print(f"[ScanOnce] src_lengths 缓存完成。")

        else:
            # 加载缓存
            if os.path.exists(self.lengths_cache):
                with open(self.lengths_cache, "r") as f:
                    self.src_lengths = json.load(f)
                assert len(self.src_lengths) == len(self.data), \
                    f"{self.lengths_cache} 与 CSV 条数不一致，请删除缓存后重建。"
            else:
                # 没有缓存就快速只算 src（不写黑名单）
                print("[Warmup] 未找到 lengths_cache，快速计算 src 时长（不写黑名单）...")
                for rec in tqdm(self.data, desc="Warmup-Lengths", unit="sample"):
                    src_audio_path = os.path.join(self.base_path, rec['audio_path'])
                    d_src = self._safe_duration(src_audio_path)
                    self.src_lengths.append(int(round(d_src * self.sr)))

    def _safe_duration(self, path: str) -> float:
        try:
            return librosa.get_duration(path=path)
        except Exception as e:
            # 出错时返回 0，后续会被分到最短桶；你也可以选择丢弃该样本
            print(f"[Warn] get_duration 失败: {path} ({e})，用 0 代替")
            return 0.0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        # 跳过黑名单
        tries = 0
        while True:
            rec = self.data[index]
            src_audio_path = os.path.join(self.base_path, rec['audio_path'])
            trg_audio_path = rec['vc_path']
            uid = f"{src_audio_path}||{trg_audio_path}"
            if uid not in self.blacklist:
                break
            index = random.randrange(len(self.data))
            tries += 1
            if tries > len(self.data):
                raise RuntimeError("No valid samples found (all are blacklisted)!")

        caption = rec["trg_instruct"]

        full_data = {
            "dataset_name": "InstructSpeech",
            "src_audio_path": src_audio_path,
            "trg_audio_path": trg_audio_path
        }

        # 读音频（到这里才真正解码）
        src_audio_data = self.load_audio_data(path=src_audio_path, type_="src")
        trg_audio_data = self.load_audio_data(path=trg_audio_path, type_="trg")
        full_data.update(src_audio_data)
        full_data.update(trg_audio_data)

        # 文本
        target = caption
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)
        full_data.update({"caption": caption, "target": target})

        return full_data

    def load_audio_data(self, path: str, type_: str) -> dict:
        audio = load(path=path, sr=self.sr)  # numpy [C, T]
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)
        # 计算 encoder 有效帧数
        valid_len = audio.shape[1] // TOTAL_STRIDE + 1
        return {
            f"{type_}_audio": torch.from_numpy(audio),                # [C, T]
            f"{type_}_audio_valid_length": torch.tensor(valid_len, dtype=torch.long),
        }


# ====== Bucketing：让同一 batch 内长度更接近 ======
class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        bucket_size_multiplier: int = 100,
        seed: int = 42,
    ):
        """
        lengths: 每个样本的长度（建议用采样点数）
        bucket_size_multiplier: 桶大小 = batch_size * bucket_size_multiplier
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_size_multiplier = bucket_size_multiplier
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        bucket_size = self.batch_size * self.bucket_size_multiplier
        for i in range(0, len(indices), bucket_size):
            bucket = indices[i:i + bucket_size]
            # 桶内按长度排序（升序）
            bucket.sort(key=lambda idx: self.lengths[idx])
            # 切成批
            batches = [bucket[j:j + self.batch_size] for j in range(0, len(bucket), self.batch_size)]
            if self.drop_last:
                batches = [b for b in batches if len(b) == self.batch_size]
            # 打乱批顺序（保持每个批内相近）
            if self.shuffle:
                rng.shuffle(batches)
            for b in batches:
                yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    dataset_name = [item["dataset_name"] for item in batch]
    src_audio_path = [item["src_audio_path"] for item in batch]
    trg_audio_path = [item["trg_audio_path"] for item in batch]
    caption = [item["caption"] for item in batch]
    target = [item["target"] for item in batch]

    # [C, T] -> [T] -> pad -> [B, max_T] -> unsqueeze(1) -> [B, 1, max_T]
    src_audio = pad_sequence(
        [item["src_audio"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0.0
    ).unsqueeze(1)

    trg_audio = pad_sequence(
        [item["trg_audio"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0.0
    ).unsqueeze(1)

    src_audio_valid_length = torch.stack([item["src_audio_valid_length"] for item in batch])
    trg_audio_valid_length = torch.stack([item["trg_audio_valid_length"] for item in batch])

    return {
        "dataset_name": dataset_name,
        "src_audio_path": src_audio_path,
        "trg_audio_path": trg_audio_path,
        "src_audio": src_audio,                                # [B, 1, max_T]
        "trg_audio": trg_audio,                                # [B, 1, max_T]
        "src_audio_valid_length": src_audio_valid_length,      # [B]
        "trg_audio_valid_length": trg_audio_valid_length,      # [B]
        "caption": caption,
        "target": target,
    }


# ====== 使用示例 ======
if __name__ == "__main__":
    ds = InstructSpeech(
        root="/path/to/your.csv",
        sr=22050,
        base_path="/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_train",
        max_duration=10.0,
        scan_once=True,
        blacklist_file="long_audio_blacklist.txt",
        lengths_cache="src_lengths.json",
        check_target=False,   # 只检查 source
    )

    sampler = BucketBatchSampler(
        lengths=ds.src_lengths,   # 采样点（已缓存）
        batch_size=8,
        drop_last=False,
        shuffle=True,
        bucket_size_multiplier=100,
        seed=1234,
    )

    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, type(v), len(v))
        break

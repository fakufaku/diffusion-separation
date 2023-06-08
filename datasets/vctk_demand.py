import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torchaudio
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from .wsj0_mix import max_collator

split_map = {
    "test",
    "train",
}


class NoisyDataset(Dataset):
    def __init__(
        self,
        audio_path: Union[str, Path],
        audio_len: Union[int, float] = 4,
        fs: Optional[int] = 16000,
        augmentation: Optional[bool] = False,
        split="train",
    ):

        ## In case of Valentini dataseti
        audio_path = Path(to_absolute_path(str(audio_path)))
        audio_path = os.path.join(audio_path, split)
        self.noisy_path = os.path.join(audio_path, "noisy")
        self.clean_path = os.path.join(audio_path, "clean")
        self.file_list = os.listdir(self.noisy_path)

        self.audio_len = audio_len * fs
        self.fs = fs
        self.aug = augmentation
        self.split = split

        if split not in split_map:
            raise ValueError(
                f"The split parameter must be 'train' or 'test' (passed {split})"
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        noisy_path = os.path.join(self.noisy_path, self.file_list[idx])
        clean_path = os.path.join(self.clean_path, self.file_list[idx])

        noisy, sr = torchaudio.load(noisy_path)
        clean, sr2 = torchaudio.load(clean_path)

        if self.split == "test":
            tgt = torch.cat([clean, noisy - clean], dim=0)
            return noisy, tgt

        ori_len = noisy.shape[-1]
        if ori_len < self.audio_len:
            noisy = torch.tile(noisy, dims=(2,))[..., : self.audio_len]
            clean = torch.tile(clean, dims=(2,))[..., : self.audio_len]
        else:
            st_idx = random.randint(0, ori_len - self.audio_len)
            noisy = noisy[..., st_idx : st_idx + self.audio_len]
            clean = clean[..., st_idx : st_idx + self.audio_len]

        if self.aug:
            noise = noisy - clean
            new_idx = torch.randperm(clean.size(0))
            noisy = noise[new_idx] + clean

        tgt = torch.cat([clean, noisy - clean], dim=0)

        return noisy, tgt


class Valentini_Module(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self.datasets = {}

    def setup(self, *args, **kwargs):

        for split in split_map:
            self.datasets[split] = instantiate(self.cfg.datamodule[split].dataset)
            if split == "train":
                tot_len = len(self.datasets["train"])
                train_len = (int)(tot_len * 0.9)
                val_len = tot_len - train_len
                self.datasets["train"], self.datasets["val"] = random_split(
                    self.datasets["train"], [train_len, val_len]
                )

    def _get_dataloader(self, split):
        return torch.utils.data.DataLoader(
            self.datasets[split],
            collate_fn=max_collator,
            **self.cfg.datamodule[split].dl_opts,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


if __name__ == "__main__":
    from scipy.io.wavfile import write as WavWrite

    noisy_path_csv = "VCTK_DEMAND/"
    data_set = NoisyDataset(
        audio_path=noisy_path_csv, audio_len=3, augmentation=True, split="train"
    )
    test_dataloader = DataLoader(
        dataset=data_set, batch_size=8, shuffle=False, sampler=None
    )
    for i, batch in enumerate(test_dataloader):
        noisy, clean = batch
        rand_idx = int((torch.rand(1) * 1000).item())
        print(clean.shape, noisy.shape)

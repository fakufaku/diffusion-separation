# 2023 (c) LINE Corporation
# MIT License
import fast_bss_eval
import torch
from pesq import pesq


class SISDRLoss(torch.nn.Module):
    def __init__(
        self, zero_mean=False, clamp_db=None, reduction="mean", sign_flip=False
    ):
        super().__init__()
        self.reduction = reduction
        self.sign_flip = sign_flip
        self.zero_mean = zero_mean
        self.clamp_db = clamp_db

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Argument reduction must be one of 'none'|'mean'|'sum'")

    def forward(self, est, ref):

        neg_si_sdr = fast_bss_eval.si_sdr_pit_loss(
            est, ref, clamp_db=self.clamp_db, zero_mean=self.zero_mean
        )

        if self.sign_flip:
            neg_si_sdr *= -1.0

        if self.reduction == "mean":
            return neg_si_sdr.mean()
        elif self.reduction == "sum":
            return neg_si_sdr.sum()
        elif self.reduction == "none":
            return neg_si_sdr
        else:
            raise ValueError("Argument reduction must be one of 'none'|'mean'|'sum'")


class PESQ(torch.nn.Module):
    def __init__(self, mode="wb", fs=16000):
        super().__init__()
        self.mode = mode
        self.fs = fs

    def forward(self, est, ref):

        est = est.cpu().numpy()
        ref = ref.cpu().numpy()

        ave_pesq = list()
        for ii in range(4):
            ave_pesq.append(pesq(self.fs, ref[ii, 0], est[ii, 0], self.mode))
        p_esq = torch.mean(torch.tensor(ave_pesq))

        return p_esq

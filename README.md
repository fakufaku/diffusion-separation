Diffusion-based Generative Speech Source Separation
===================================================

This repository contains the code to reproduce the results of the paper [Diffusion-based Generative Speech
Source Separation](https://arxiv.org/abs/2210.17327) presented at ICASSP 2023.

We propose DiffSep, a new single channel source separation method based on
score-matching of a stochastic differential equation (SDE). We craft a tailored
continuous time diffusion-mixing process starting from the separated sources
and converging to a Gaussian distribution centered on their mixture. This
formulation lets us apply the machinery of score-based generative modelling.
First, we train a neural network to approximate the score function of the
marginal probabilities or the diffusion-mixing process. Then, we use it to
solve the reverse time SDE that progressively separates the sources starting
from their mixture. We propose a modified training strategy to handle model
mismatch and source permutation ambiguity. Experiments on the WSJ0 2mix dataset
demonstrate the potential of the method. Furthermore, the method is also
suitable for speech enhancement and shows performance competitive with prior
work on the VoiceBank-DEMAND dataset.

Show Me How to Separate Wav Files!
----------------------------------

We got you covered. Just run the following command (after setting up the environment as described under Training).
```bash
python separate.py path/to/wavfiles/folder path/to/output/folder
```
where `path/to/wavfiles/folder` points to a folder containing wavfiles. The
input files should be sampled at `8 kHz` for the default model. Two speakers
are separated and stored in `path/to/output/folder/s1` and
`path/to/output/folder/s2`, respectively.
The model weights are stored on [huggingface](https://huggingface.co/fakufaku/diffsep).


Configuration
-------------

Configuration is done using the [hydra](https://hydra.cc/docs/intro/) hierarchical configuration package.
The hierarchy is as follows.
```
config/
|-- config.yaml  # main config file
|-- datamodule  # config of dataset and dataloaders
|   |-- default.yaml
|   `-- diffuse.yaml  # smaller batch size for CDiffuse
|-- model
|   |-- default.yaml  # NCSN++ model
|   `-- diffuse.yaml  # CDiffuse model
`-- trainer
    `-- default.yaml  # config of pytorch-lightning trainer
```

Dataset
-------

The `wsj0_mix` dataset is expected in `data/wsj0_mix`
```
data/wsj0_mix/
|-- 2speakers
|   |-- wav16k
|   |   |-- max
|   |   |   |-- cv
|   |   |   |-- tr
|   |   |   `-- tt
|   |   `-- min
|   |       |-- cv
|   |       |-- tr
|   |       `-- tt
|   `-- wav8k
|       |-- max
|       |   |-- cv
|       |   |-- tr
|       |   `-- tt
|       `-- min
|           |-- cv
|           |-- tr
|           `-- tt
`-- 3speakers
    |-- wav16k
    |   `-- max
    |       |-- cv
    |       |-- tr
    |       `-- tt
    `-- wav8k
        `-- max
            |-- cv
            |-- tr
            `-- tt
```

The `VCTK-DEMAND` dataset is expected in `data/VCTK_DEMAND`

```
data/VCTK_DEMAND/
|--train
|   |-- noisy
|   `-- clean
`-- test
    |-- noisy
    `-- clean
```

Training
--------

Preparation
```bash
conda env create -f environment.yaml
conda activate diff-sep
```
Run training. The results of training and tensorboard files are stored in `./exp/`.
```bash
python ./train.py
```
Thanks to hydra, parameters can be added easily
```bash
python ./train.py model.sde.sigma_min=0.1
```

The training can be run in **multi-gpu** setting by overriding the trainer config
`trainer=allgpus`.  Since validation is quite expensive to do, we set
`trainer.check_val_every_n_epoch=5` to run it only every 5 epochs.
The train and validation batch sizes are multiplied by the number of GPUS.

Evaluation
----------

The `evaluation.py` script can be used to run the inference for `val` and `test` datasets.
```bash
$ python ./evaluate.py --help
usage: evaluate.py [-h] [-d DEVICE] [-l LIMIT] [--save-n SAVE_N] [--val] [--test] [-N N] [--snr SNR] [--corrector-steps CORRECTOR_STEPS] [--denoise DENOISE] ckpt

Run evaluation on validation or test dataset

positional arguments:
  ckpt                  Path to checkpoint to use

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device to use (default: cuda:0)
  -l LIMIT, --limit LIMIT
                        Limit the number of samples to process
  --save-n SAVE_N       Save a limited number of output samples
  --val                 Run on validation dataset
  --test                Run on test dataset
  -N N                  Number of steps
  --snr SNR             Step size of corrector
  --corrector-steps CORRECTOR_STEPS
                        Number of corrector steps
  --denoise DENOISE     Use denoising in solver
  --enhance             Run evaluation for speech enhancement task (default: false)
```
This will save the results in a folder named `results/{exp_name}_{ckpt_name}_{infer_params}`.
The option `--save-n N` allows to save the firs `N` samples as figures and audio samples.

Reproduce
---------

### Separation

```shell
# train
python ./train.py experiment=icassp-separation

# evaluate
python ./evaluate_mp.py exp/default/<YYYY-MM-DD_hh-mm-ss>_experiment-icassp-separation/checkpoints/epoch-<NNN>_si_sdr-<F.FFF>.ckpt --split test libri-clean
```

### Enhancement

```shell
# train
python ./train.py experiment=noise-reduction

# evaluate
python ./evaluate.py exp/enhancement/<YYYY-MM-DD_hh-mm-ss>_experiment-noise-reduction/checkpoints/epoch-<NNN>_si_sdr-<F.FFF>.ckpt --test --pesq-mode wb
```

License
-------

2023 (c) LINE Corporation

The repo is released under MIT license, but please refer to individual files for their specific license.

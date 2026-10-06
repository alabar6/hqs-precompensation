# Half-Quadratic Splitting Method in Image Precompensation

The code for training and testing a precompensation algorithm based on the Half-Quadratic Splitting method (HQS) is provided here.

# How to use this repo?


Before starting, I recommend to set virtual environment:

```bash
python -m venv env
pip install -r requirements.txt
```

## Data Preparation

If you just want to see some examples of work, a small piece of the SCA-2023 dataset is provided in the `dataset` folder.

## Training

After preparing data, start training:

```bash
python -m train \
    --config_path data/configs/train_config.json \
    --save_path ./results \
    --seed 12345 \
    --verbose True \
    --device cuda
```

where:
- `config-path` is a path to train config,
- `save-path` is a path where train artifacts will be saved,
- `seed` is a random seed,
- `verbose` is a flag; if True, shows progress bar,
- `device` is a device.

After running a script, 

## Test

After preparing data, start training:

```bash
python -m test \
    --config_path data/configs/train_config.json \
    --save_path \
    --seed 12345 \
    --verbose True \
    --device cuda
```

where:
- `config-path` is a path to train config,
- `save-path` is a path where test results will be saved,
- `seed` is a random seed,
- `verbose` is a flag; if True, shows progress bar,
- `device` is a device.




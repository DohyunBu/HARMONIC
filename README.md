# Harmonized Cone for Feasible and Non-conflict Directions in Training Physics-Informed Neural Networks

This repository provides the implementation of the paper **"[Harmonized Cone for Feasible and Non-conflict Directions in Training Physics-Informed Neural Networks](https://iclr.cc/virtual/2026/poster/10009677)"**.


---

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   * [Log Output](#log-output)
   * [Parameter Settings](#parameter-settings)
3. [Dependencies](#dependencies)
4. [Benchmark](#benchmark)

---

## Installation

Clone this repository and install required packages:

```bash
python -m pip install -r requirements.txt
```

---

## Usage

Run all 5 cases with default settings:

```shell
python benchmark.py [--name EXP_NAME]
```

Make sure `runs/` directory exists before running.

### Log Output

After each run the `./runs/{date}-{time}-{name}/{benchmark_idx}-{iter}/log.txt` will contain both training progress and your final test‐set metrics. A typical snippet looks like:

```
***** Begin #0-0 *****
Warning: 8192 points required, but 8281 points sampled.
Compiling model...
'compile' took 0.000072 s

PDE Class Name: Wave1D
Training model...

Step      Train loss                                  Test loss                                   Test metric
0         [3.25e-01, 6.20e-02, 1.10e+00, 1.19e-01]    [3.28e-01, 6.20e-02, 1.10e+00, 1.19e-01]    []  
100       [2.43e-03, 1.75e-03, 2.65e-01, 1.08e-01]    [1.86e-03, 1.75e-03, 2.65e-01, 1.08e-01]    []  
Validation: epoch 100 MSE 0.43996 MAE 0.53997 MXE 1.76786 L1RE 1.17441 L2RE 1.18678 CRMSE 0.33817
Unweighted Loss: [2.43e-03, 1.75e-03, 2.65e-01, 1.08e-01]  [1.86e-03, 1.75e-03, 2.65e-01, 1.08e-01] Weights: [1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00]
...
50000     [1.23e-02, 4.89e-05, 2.12e-05, 1.98e-04]    [1.34e-02, 4.89e-05, 2.12e-05, 1.98e-04]    []  
Validation: epoch 50000 MSE 0.00063 MAE 0.01808 MXE 0.09930 L1RE 0.03932 L2RE 0.04494 CRMSE 0.00005 FRMSE (0.00968, 0.00593, 0.00053)
Unweighted Loss: [1.23e-02, 4.89e-05, 2.12e-05, 1.98e-04]  [1.34e-02, 4.89e-05, 2.12e-05, 1.98e-04] Weights: [1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00]
```
- **`Validation: epoch 50000 ... L2RE`** reports your model’s performance on the test set at epoch 50000.

### Parameter Settings

You can customize training via command-line arguments. The available options are:

```bash
--name 
--device
--hidden-layers
--lr
--iter
--log-every
--trainseed
--method
```

Example:

```shell
python benchmark.py --name EXP_NAME --device 0 --hidden-layers 50*3 --lr 1e-3 --iter 50000 --log-every 100 --trainseed [2025,9,19,98,220621] --method harmonic
```

---

## Dependencies

Tested on **Windows 11**, **Python 3.9.21**.

```
dill==0.3.6
matplotlib==3.7.1
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
scikit-optimize==0.10.2
scipy==1.11.4
pycddlib==3.0.2
torch==2.5.1+cu118
torchaudio==2.5.1+cu118
torchvision==0.20.1+cu118
```

---

## Benchmark

* **PINNacle** (excluding Volterra): Download via `https://github.com/i207M/PINNacle`.
* **A-PINN**: Using IDE via `https://github.com/YUANLei2021/A-PINN`.


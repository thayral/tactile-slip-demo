# Tactile slip detection demo (PzE sensors)

Minimal demo for slip detection in robotic manipulation using tactile PzE signals (FFT + GRU pipeline).

## Run

```bash
# 1) Create + activate venv
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run inference + plots
python demo_infer_clean.py
```


<p align="center">
  <img src="media/exp_20240306_202240_run_20240306_202613.png" width="48%" />
  <img src="media/exp_20240312_210628_run_20240312_211137.png" width="48%" />
</p>




The repo is about 30Mo.  
This includes a small demo dataset for inference.  
(It takes 5 min to download, install dependencies and run.)   


Runs on CPU by default (CUDA not required).



> Design note: The dataset is split into two files (inference data and additional visualization data). 
> Raw signals used for visualization are loaded separately from the inference dataset
> and matched via sample identifiers, to keep the model pipeline side-effect free.

---


➡️ Project page: [https://thayral.github.io/tactile-slip-detection-pze/](https://thayral.github.io/tactile-slip-detection-pze/)




➡️ This work is part of the PhD thesis  
**Learning-based slip detection for adaptive grasp control**  
Théo AYRAL -
CEA (Leti & List) · Université Paris-Saclay


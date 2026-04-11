# EE7207 Assignment 2: Knowledge Distillation for Cryptocurrency Sentiment Analysis

> **Important Note on OS:** This setup was validated on **native Ubuntu 22.04** (dual boot, not WSL).

## System Requirements

* **OS:** Ubuntu 22.04 (Native installation)
* **GPU:** NVIDIA GPU (Tested on RTX 4060)
* **RAM:** 8 GB recommended
* **Storage:** ~50 GB free disk space recommended

---

### 1. Clone the repository:

Clone the repository:

```bash
git clone https://github.com/MysEcho/EE7207_assignment.git
cd assignment_2
```

### 2. Run your code

Assuming the Pixi environment has been set up and the user is inside the Pixi Shell.

To run the inference script:

```bash
python inference.py
```

To train your own model:

```bash
python train.py
```

The generated synthetic datasets are released and can be found in /data/generated_datasets. The model checkpoints can be found in /checkpoints.

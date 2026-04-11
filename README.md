# EE7207 Assignments

> **Important Note on OS:** All codebases were validated on **native Ubuntu 22.04** (dual boot, not WSL).

## System Requirements

* **OS:** Ubuntu 22.04 (Native installation)
* **GPU:** NVIDIA GPU (Tested on RTX 4060)
* **RAM:** 8 GB recommended
* **Storage:** ~50 GB free disk space recommended

---

## Setup Guide

### 1. NVIDIA Driver Installation(Ignore if you already have it installed)

First, check your available drivers:
```bash
ubuntu-drivers devices
```

Install the recommended driver and reboot your system:

```bash
sudo apt install nvidia-driver-590-open
sudo reboot
```

Verify the installation:

```bash
nvidia-smi
```
Also make sure you install the CUDA Toolkit.

### 2. Installe Pixi 

We use Pixi for deterministic package management. Install it via curl:

```bash
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash
```

### 3. Setup the environment

From the project root, run this command:

```bash
pixi install
```

Then activate the Pixi environment:

```bash
pixi shell
```

For further instructions, please navigate to the specific assignment folders.
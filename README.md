# [BETA] Side-Step for ACE-Step 1.5

```bash
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
  by dernet     ((BETA TESTING))
```

**Side-Step** is a high-performance training "sidecar" for [ACE-Step 1.5](https://github.com/TencentGameMate/ACE-Step). It provides a corrected LoRA fine-tuning implementation that fixes fundamental bugs in the original trainer while adding low-VRAM support for local GPUs.

## 🚀 Why Side-Step?

The original ACE-Step trainer has two critical discrepancies from how the base models were actually trained. Side-Step was built to bridge this gap:

1.  **Continuous Timestep Sampling:** The original trainer uses a discrete 8-step schedule. Real music isn't discrete. Side-Step implements **Logit-Normal continuous sampling**, ensuring the model learns the full range of the denoising process.
2.  **CFG Dropout (Classifier-Free Guidance):** The original trainer lacks condition dropout. Side-Step implements a **15% null-condition dropout**, teaching the model how to handle both prompted and unprompted generation. Without this, inference quality suffers.
3.  **Non-Destructive Architecture:** It lives *alongside* your ACE-Step installation. It imports what it needs without touching a single line of the original source code.
4.  **Built for the cloud** The original Gradio breaks when you try to use it for training. Use this instead :)

---

## ⚠️ Beta Status & Support
**Current Version:** 0.2.0-beta

| Feature | Status | Note |
| :--- | :--- | :--- |
| **Fixed Training** | ✅ Working | Recommended for all users. |
| **Vanilla Training** | ✅ Working | For reproduction of old results. |
| **Interactive Wizard** | ✅ Working | `python train.py` with no args. |
| **TUI (Textual UI)** | ❌ **BROKEN** | Do not use `sidestep_tui.py` yet. |
| **CLI Preprocessing** | ❌ Planned | Use the Gradio UI for preprocessing for now. |
| **Gradient Estimation** | ❌ Planned | Coming in future update. |

---

## 📦 Installation

Side-Step is designed to be placed **inside** your existing ACE-Step 1.5 folder.

1. **Requirements:** Ensure you have ACE-Step 1.5 installed and working.
2. **Clone into ACE-Step:**
   ```bash
   cd path/to/ACE-Step
   git clone https://github.com/koda-dernet/Side-Step.git temp_side
   mv temp_side/* .
   rm -rf temp_side
   ```
3. **Install Dependencies:**
   (We recommend [uv](https://github.com/astral-sh/uv) for 10x faster installation and syncronization with the actual ACE-Step 1.5 project)
   ```bash
   # Using uv
   uv pip install -r requirements-sidestep.txt
   
   # Using standard pip
   pip install -r requirements-sidestep.txt
   ```
4. **Optional (Low VRAM):**
   ```bash
   uv pip install bitsandbytes>=0.45.0
   ```
5. **Optional (For bragging rights by using the Prodigy optimizer)**
   ```bash
   uv pip intall prodigyopt>=1.1.2
    ```
---

## 🛠️ Usage

### Option A: The Interactive Wizard (Recommended)
Simply run the script with no arguments. It will ask you everything it needs to know.
```bash
python train.py
```

### Option B: The Quick Start One-Liner
If you have your preprocessed tensors ready in `./my_data`, run:
```bash
python train.py fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./my_data \
    --output-dir ./output/my_lora \
    --epochs 100
```

---

## 📉 Optimization & VRAM Profiles
Side-Step is optimized for both heavy Cloud GPUs (H100/A100) and local "underpowered" gear (RTX 3060/4070).

| Profile | VRAM | Key Settings |
| :--- | :--- | :--- |
| **Comfortable** | 24GB+ | Standard AdamW, Batch 2+ |
| **Standard** | 16-24GB | Standard AdamW, Batch 1 |
| **Tight** | 10-16GB | **AdamW8bit**, Grad Checkpointing, Encoder Offloading |
| **Minimal** | <10GB | **AdaFactor** or **AdamW8bit**, Rank 16, High Grad Accumulation |

### Pro Features:
*   **`--offload-encoder`**: Moves the heavy VAE and Text Encoders to CPU after setup. Frees ~4GB VRAM.
*   **`--gradient-checkpointing`**: Drastically reduces memory usage during the backward pass.
*   **`--optimizer-type prodigy`**: Uses the Prodigy optimizer to automatically find the best learning rate for you.

---

## 📂 Project Structure
```text
.
├── train.py                 <-- Your main entry point
├── requirements-sidestep.txt
└── acestep/
    ├── training/            <-- Original ACE-Step code (Untouched)
    └── training_v2/         <-- Side-Step logic
        ├── trainer_fixed.py <-- The corrected logic
        ├── optim.py         <-- 8-bit and adaptive optimizers
        └── ui/              <-- Wizard and CLI visual logic
```

---

## 🤝 Contributing
Contributions are welcome! Specifically looking for help fixing the **Textual TUI** and completing the **CLI Preprocessing** module.

**License:** Follows the original ACE-Step 1.5 licensing

# llm-from-scratch

A character-level GPT built from scratch, following Andrej Karpathy's
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture).

## Requirements

- Python 3.9–3.13
- pip

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

Install PyTorch:

```bash
pip install torch
```

For platform-specific installs (CUDA, older versions, etc.), see
https://pytorch.org/get-started/locally/

Verify the install:

```bash
python -c "import torch; print(torch.__version__)"
```

## Run

```bash
python gpt-complete.py
```


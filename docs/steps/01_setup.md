# Step 1: Setup

**Goal:** a project that anyone can rebuild on their own computer.

## What we did
- Worked on a Mac in VS Code (GitHub Codespaces ran out of storage, so we moved to local).
- Installed **Python 3.12.10** from python.org and made a private environment called `.venv`.
- Made the folders: `data/`, `src/exoplanet/`, `configs/`, `results/`, `docs/`, `tests/`.
- Wrote `requirements.txt` (the package list) and `.gitignore` (files git must not upload).

## Commands
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to check it worked
- The terminal line starts with `(.venv)`.
- `python --version` prints `Python 3.12.x`.

## Why we did it this way
| Choice | Reason |
|---|---|
| Virtual environment | Keeps this project's packages separate from everything else on the computer. |
| Python 3.12 | TensorFlow supports it and python.org has a Mac installer for 3.12.10. Python 3.15 exists, but we could not confirm TensorFlow works on it. Newer 3.11 releases have no Mac installer. |
| `.gitignore` | The data files are huge (GitHub rejects files over 100 MB) and the `.venv` should never be uploaded. |
| Plain `.py` scripts | They run the same way every time and are easy to publish. |

## Problems we hit
| Problem | Fix |
|---|---|
| `zsh: command not found: pip` | `pip` only exists inside an active `.venv`. Activate it first. |
| Mac showed Python 3.9 | That is the old system Python. The new one is run as `python3.12`. |

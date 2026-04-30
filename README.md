# Setup Instructions

**1. Install `uv`** 

(macOS/Linux)
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

(windows)
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

**2. Create a virtual environment and install dependencies**
```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```
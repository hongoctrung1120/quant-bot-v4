# Python 3.10 setup

This project is pinned for Python 3.10.x and the requested pip 22.3.1 environment.

## Windows PowerShell

If PowerShell blocks `Activate.ps1`, activation is not required. Run the virtual-environment Python directly.

```powershell
cd "D:\Bot quant\quant_bot_v4_complete\qbwork"

# Create environment with the installed Python 3.10
py -3.10 -m venv .venv

# Confirm runtime
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version

# Optional: ensure the requested pip version
.\.venv\Scripts\python.exe -m pip install pip==22.3.1

# Install pinned dependencies
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt

# Run tests
.\.venv\Scripts\python.exe -m pytest -q

# Run sample Dollar-Bar backtest
.\.venv\Scripts\python.exe main.py --backtest
```

## Important

Do not install the latest NumPy on this environment. NumPy is pinned to `1.26.4` because it provides Windows wheels for Python 3.10 and keeps the scientific stack stable for this project.

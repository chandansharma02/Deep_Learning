# Windows Setup (CPU Only)

1. Install [Python 3.10+](https://www.python.org/downloads/windows/).
2. Open *PowerShell* in the project root.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\activate
   ```
4. Install packages:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Generate synthetic data:
   ```powershell
   python .\data\make_synthetic_data.py
   ```
6. Launch notebooks:
   ```powershell
   jupyter notebook
   ```
All code runs on CPU; no GPU or CUDA required.

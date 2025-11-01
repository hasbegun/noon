# Conda Setup Guide

This project uses conda for environment management with the environment name `noon2`.

## Quick Start with Conda

```bash
cd ml

# One-command setup
make install

# Activate environment
conda activate noon2

# Verify installation
python scripts/verify_installation.py
```

## Manual Conda Setup

If you prefer manual setup:

```bash
# 1. Create conda environment
conda create -n noon2 python=3.11 -y

# 2. Activate environment
conda activate noon2

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install SAM2
mkdir -p .tmp
cd .tmp
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ../..

# 5. Setup environment
cp .env.example .env
```

## Conda Commands

### Environment Management

```bash
# List all conda environments
conda env list

# Activate environment
conda activate noon2

# Deactivate environment
conda deactivate

# Show environment info
conda info --envs

# List installed packages
conda list
```

### Using the Environment

```bash
# All make commands automatically use the noon2 environment
make train
make serve
make inference IMAGE=food.jpg

# Or use conda run for one-off commands
conda run -n noon2 python scripts/train.py

# Or activate and run
conda activate noon2
python scripts/train.py
```

## Why Conda?

1. **Better Dependency Management**: Conda handles both Python and system dependencies
2. **Isolated Environments**: Complete isolation from system Python
3. **Cross-platform**: Works on macOS, Linux, and Windows
4. **Apple Silicon Support**: Optimized for M1/M2/M3/M4 Macs
5. **Reproducibility**: Easy to share and recreate environments

## Conda vs pip

This project uses conda for environment management but pip for package installation:

- **Conda**: Creates and manages the `noon2` environment
- **pip**: Installs Python packages (PyTorch, FastAPI, etc.)

This hybrid approach provides the best of both worlds.

## Exporting Environment

```bash
# Export environment to file
conda env export > environment.yml

# Create from exported file
conda env create -f environment.yml
```

## Troubleshooting

### Conda not found

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Or on Intel Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### Environment activation fails

```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.

# Restart shell
exec bash
```

### Wrong Python version

```bash
# Remove and recreate environment
conda env remove -n noon2
conda create -n noon2 python=3.11 -y
conda activate noon2
make install-deps
```

## Advanced: Multi-Machine Setup

For distributed training across multiple machines, each machine needs:

```bash
# On each machine:
# 1. Install conda
# 2. Clone repository
# 3. Run setup
cd ml
make install
conda activate noon2

# Then run distributed training (see README for details)
```

## Cleanup

```bash
# Remove environment completely
make purge

# Or manual removal
conda env remove -n noon2 -y
```

## Tips

1. **Always activate** before running scripts manually:
   ```bash
   conda activate noon2
   python scripts/train.py
   ```

2. **Use Make** for automatic environment handling:
   ```bash
   make train  # Automatically uses noon2
   ```

3. **Check status** anytime:
   ```bash
   make status
   make conda-info
   ```

4. **Keep updated**:
   ```bash
   conda activate noon2
   pip install --upgrade pip
   pip install -r requirements.txt --upgrade
   ```

# Get Started in 3 Steps

## Prerequisites
- Conda installed (Anaconda or Miniconda)
- Git installed
- macOS with Apple Silicon (M3/M4) recommended

## Step 1: Install (5 minutes)

```bash
cd /Users/innox/projects/noon2/ml
make install
```

This will:
- ✅ Check conda installation
- ✅ Create conda environment `noon2` with Python 3.11
- ✅ Install all Python dependencies
- ✅ Clone and install SAM2 from GitHub
- ✅ Create .env configuration file

## Step 2: Activate & Verify (1 minute)

```bash
# Activate conda environment
conda activate noon2

# Verify installation
python scripts/verify_installation.py

# Check system status
make status
```

## Step 3: Start Using! (Now)

Choose your workflow:

### Option A: Run Demo
```bash
make demo
```

### Option B: Start API Server
```bash
make serve
```

### Option C: Preprocess & Train
```bash
make preprocess
make train-quick
```

### Option D: Analyze Image
```bash
make inference IMAGE=path/to/food.jpg LABELS="rice,chicken,salad"
```

## Need Help?

```bash
# See all available commands
make help

# Check system status
make status

# Get environment info
make conda-info

# Read docs
cat README.md
cat QUICKSTART.md
```

## Common Issues

### "conda: command not found"
Install Miniconda:
```bash
# For Apple Silicon (M1/M2/M3/M4)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### SAM2 installation fails
The system will use a placeholder model. To install manually:
```bash
conda activate noon2
make install-sam2
```

### Port 8000 already in use
Change port:
```bash
make serve PORT=8080
```

## What's Next?

1. Read [QUICKSTART.md](QUICKSTART.md) for detailed guide
2. Read [README.md](README.md) for complete documentation
3. Read [CONDA_SETUP.md](CONDA_SETUP.md) for conda details
4. Run `make help` to see all commands

## Quick Reference

| Task | Command |
|------|---------|
| Install | `make install` |
| Activate | `conda activate noon2` |
| Train | `make train` |
| Inference | `make inference IMAGE=...` |
| API Server | `make serve` |
| Help | `make help` |
| Status | `make status` |
| Clean | `make clean` |

---

**That's it! You're ready to detect food!** 🎉

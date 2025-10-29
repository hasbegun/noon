#!/usr/bin/env python3
"""
Verify installation and dependencies
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check if all required packages can be imported"""
    print("Checking package imports...")

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "scipy": "SciPy",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic",
        "httpx": "HTTPX",
        "pandas": "Pandas",
        "albumentations": "Albumentations",
        "tqdm": "tqdm",
        "yaml": "PyYAML",
        "loguru": "Loguru",
        "sqlalchemy": "SQLAlchemy",
    }

    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            results[name] = True
            print(f"  ✓ {name}")
        except ImportError as e:
            results[name] = False
            print(f"  ✗ {name} - {e}")

    # Check SAM2 separately
    print("\nChecking SAM2...")
    try:
        from sam2.build_sam import build_sam2
        results["SAM2"] = True
        print("  ✓ SAM2 (installed)")
    except ImportError:
        results["SAM2"] = False
        print("  ✗ SAM2 (not installed - will use placeholder)")
        print("    Install with: make install-sam2")

    return results


def check_torch_backend():
    """Check PyTorch backend availability"""
    print("\nChecking PyTorch backends...")

    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("  ✗ CUDA not available")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) available")
        else:
            print("  ✗ MPS not available")

        # CPU is always available
        print("  ✓ CPU available")

    except Exception as e:
        print(f"  Error checking backends: {e}")


def check_project_structure():
    """Check project directory structure"""
    print("\nChecking project structure...")

    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/services",
        "src/training",
        "src/api",
        "scripts",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")


def check_data():
    """Check data directories"""
    print("\nChecking data directories...")

    data_checks = {
        "data/raw": "Raw datasets",
        "data/usda": "USDA nutrition data",
        "data/processed": "Processed data",
        "models/pretrained": "Pretrained models",
        "models/segmentation": "Trained models",
    }

    for path_str, desc in data_checks.items():
        path = Path(path_str)
        if path.exists():
            if path.is_symlink():
                target = path.resolve()
                print(f"  ✓ {desc} → {target}")
            else:
                print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} (not found)")


def check_config():
    """Check configuration"""
    print("\nChecking configuration...")

    try:
        from src.config import config

        print(f"  ✓ Configuration loaded")
        print(f"    Device: {config.device}")
        print(f"    Batch size: {config.batch_size}")
        print(f"    Image size: {config.image_size}")
        print(f"    Data path: {config.data_root}")
        print(f"    Models path: {config.models_root}")

    except Exception as e:
        print(f"  ✗ Configuration error: {e}")


def main():
    """Main verification function"""
    print("=" * 60)
    print("Food Detection & Nutrition Analysis - Installation Verification")
    print("=" * 60)
    print()

    # Run checks
    results = check_imports()
    check_torch_backend()
    check_project_structure()
    check_data()
    check_config()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"Total packages: {total}")
    print(f"  ✓ Installed: {passed}")
    print(f"  ✗ Missing: {failed}")

    if failed == 0:
        print("\n✓ All checks passed! System is ready.")
        return 0
    elif results.get("SAM2", False) is False and failed == 1:
        print("\n⚠ SAM2 not installed but system will work with placeholder.")
        print("  Install SAM2 for production use: make install-sam2")
        return 0
    else:
        print("\n✗ Some packages are missing. Please install dependencies:")
        print("  make install")
        return 1


if __name__ == "__main__":
    sys.exit(main())

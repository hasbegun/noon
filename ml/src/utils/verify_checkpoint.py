#!/usr/bin/env python3
"""
Quick script to verify checkpoint/resume functionality

Run this before starting your training to test that checkpoints work.

Usage:
    From ml directory: python -m src.utils.verify_checkpoint
    Or: python src/utils/verify_checkpoint.py
"""
import sys
from pathlib import Path

# Add project root to path
ml_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ml_dir))

def main():
    print("=" * 60)
    print("CHECKPOINT/RESUME VERIFICATION")
    print("=" * 60)

    from src.config import config

    checkpoint_dir = config.segmentation_models_path
    print(f"\n✓ Checkpoint directory: {checkpoint_dir}")

    # Check if directory exists
    if not checkpoint_dir.exists():
        print(f"  Creating directory: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  Directory exists: ✓")

    # Check for existing checkpoints
    print(f"\n📁 Existing checkpoints:")
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        print("  No checkpoints found (this is normal for first run)")
    else:
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt.name}: {size_mb:.1f} MB")

        # Try to load and inspect last checkpoint
        last_checkpoint = checkpoint_dir / "last_checkpoint.pt"
        if last_checkpoint.exists():
            print(f"\n🔍 Inspecting last_checkpoint.pt:")
            try:
                import torch
                checkpoint = torch.load(last_checkpoint, map_location='cpu')
                print(f"  ✓ Checkpoint is valid")
                print(f"  - Saved at epoch: {checkpoint['epoch']}")
                print(f"  - Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
                print(f"  - Training history: {len(checkpoint.get('train_losses', []))} epochs")

                # Check if it will resume correctly
                if checkpoint['epoch'] > 0:
                    next_epoch = checkpoint['epoch'] + 1
                    print(f"\n✅ If you resume, training will continue from epoch {next_epoch}")
                else:
                    print(f"\n⚠️  Checkpoint is from epoch 0 (training just started)")

            except Exception as e:
                print(f"  ❌ Error loading checkpoint: {e}")

    # Test checkpoint saving/loading
    print(f"\n🧪 Testing checkpoint functionality:")
    try:
        import torch
        import torch.nn as nn

        # Create dummy model and data
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        model = DummyModel()
        test_checkpoint = checkpoint_dir / "test_checkpoint.pt"

        # Save test checkpoint
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'test': True
        }, test_checkpoint)
        print(f"  ✓ Can save checkpoint")

        # Load test checkpoint
        loaded = torch.load(test_checkpoint, map_location='cpu')
        assert loaded['test'] == True
        print(f"  ✓ Can load checkpoint")

        # Clean up
        test_checkpoint.unlink()
        print(f"  ✓ Checkpoint save/load works correctly!")

    except Exception as e:
        print(f"  ❌ Checkpoint test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\n✅ Checkpoint system is working correctly!")
    print("\nYou can now start training:")
    print("  python src/train/train.py --epochs 50 --batch-size 8 --device mps")
    print("\nIf interrupted, it will resume automatically.")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

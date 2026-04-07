import torch
import sys
sys.path.insert(0, '.')

print("Testing model.py return_video_features modification...")

# Create a minimal test to verify the signature change
from wan_va.modules.model import WanTransformer3DModel

# Check that forward method accepts return_video_features parameter
import inspect
sig = inspect.signature(WanTransformer3DModel.forward)
params = list(sig.parameters.keys())

print(f"Forward method parameters: {params}")

if 'return_video_features' in params:
    print("✓ return_video_features parameter added successfully")
else:
    print("✗ return_video_features parameter NOT found")
    sys.exit(1)

print("\n✅ Model modification test passed!")

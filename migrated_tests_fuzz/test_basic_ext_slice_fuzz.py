# Fuzzing tests generated from test_basic_ext_slice.py
# Original test function: test_basic_ext_slice_pt
# Generated 1 variants with mutated inputs

import torch
import numpy as np

def test_basic_ext_slice_pt_fuzz_0():
    import torch

    

# Run all fuzz variants
if __name__ == "__main__":
    variants = [test_basic_ext_slice_pt_fuzz_0]
    
    for variant_func in variants:
        try:
            print(f"Running {variant_func.__name__}...")
            variant_func()
            print(f"  ✓ {variant_func.__name__} PASSED")
        except Exception as e:
            print(f"  ✗ {variant_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

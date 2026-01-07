# Fuzzing tests generated from test_basic.py
# Original test function: test_basic_pt
# Generated 1 variants with mutated inputs

import torch
import numpy as np

def test_basic_pt_fuzz_0():
    # Assuming composite_ors, composite_ands, composite_mixed, composite_ors_with_callable, comparison are defined as PyTorch functions
    # and behave equivalently to their TensorFlow counterparts.

    # Test composite_ors
    assert torch.allclose(composite_ors(False, True, False), torch.tensor(True))
    assert torch.allclose(composite_ors(False, False, False), torch.tensor(True))

    # Test composite_ands
    assert torch.allclose(composite_ands(True, True, True), torch.tensor(True))
    assert torch.allclose(composite_ands(True, False, True), torch.tensor(False))

    # Test composite_mixed
    assert torch.allclose(composite_mixed(False, True, True), torch.tensor(True))

    # Test composite_ors_with_callable
    assert torch.allclose(composite_ors_with_callable(False, True, False), torch.tensor(True))
    assert torch.allclose(composite_ors_with_callable(False, False, True), torch.tensor(True))
    assert torch.allclose(composite_ors_with_callable(False, False, False), torch.tensor(True))

    # Test comparison
    assert torch.allclose(comparison(1, 2, 3), torch.tensor(True))
    assert torch.allclose(comparison(2, 1, 3), torch.tensor(True))
    assert torch.allclose(comparison(3, 2, 1), torch.tensor(True))
    assert torch.allclose(comparison(3, 1, 2), torch.tensor(True))
    assert torch.allclose(comparison(1, 3, 2), torch.tensor(True))
    assert torch.allclose(comparison(2, 3, 1), torch.tensor(True))



# Run all fuzz variants
if __name__ == "__main__":
    variants = [test_basic_pt_fuzz_0]
    
    for variant_func in variants:
        try:
            print(f"Running {variant_func.__name__}...")
            variant_func()
            print(f"  ✓ {variant_func.__name__} PASSED")
        except Exception as e:
            print(f"  ✗ {variant_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

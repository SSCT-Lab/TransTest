# Fuzzing tests generated from test_while_with_variable_shape_growing_vector.py
# Original test function: test_while_with_variable_shape_growing_vector_pt
# Generated 1 variants with mutated inputs

import torch
import numpy as np

def test_while_with_variable_shape_growing_vector_pt_fuzz_0(self, n, type_):
    n = type_(n)
    # Simulate the behavior of while_with_variable_shape_growing_vector using PyTorch
    result = []
    i = 1
    while i < n:
        # Simulate growing vector: append a scalar tensor at each step
        item = torch.tensor([i], dtype=torch.float32)
        result.append(item)
        i += 1

    # Concatenate all tensors along dimension 1
    final_result = torch.cat(result, dim=1) if result else torch.empty(1, dtype=torch.float32)

    # Print shape and values for debugging
    print(f"Final result shape: {final_result.shape}")
    print(f"Final result values: {final_result}")

    # Assert that the result is as expected (e.g., [1, 1, 2, ..., n-1])
    expected = torch.arange(n, dtype=torch.float32)
    assert torch.allclose(final_result, expected), f"Expected {expected}, but got {final_result}"



# Run all fuzz variants
if __name__ == "__main__":
    variants = [test_while_with_variable_shape_growing_vector_pt_fuzz_0]
    
    for variant_func in variants:
        try:
            print(f"Running {variant_func.__name__}...")
            variant_func()
            print(f"  ✓ {variant_func.__name__} PASSED")
        except Exception as e:
            print(f"  ✗ {variant_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

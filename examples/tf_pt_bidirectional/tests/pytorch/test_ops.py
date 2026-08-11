import torch


def test_relu_numerical_boundary():
    values = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
    actual = torch.relu(values)
    expected = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_sum_numerical():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    actual = torch.sum(values, dim=1)
    expected = torch.tensor([3.0, 7.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

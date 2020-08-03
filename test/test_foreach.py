import torch
import torch.cuda
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

devices = (torch.device('cpu'), torch.device('cuda:0'))

class TestForeach(TestCase):
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_same_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20
        tensors = []
        for _ in range(N):
            tensors.append(torch.zeros(H, W, device=device, dtype=dtype))

        res = torch._foreach_add(tensors, 1)
        for t in res:
            self.assertEqual(t, torch.ones(H, W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_different_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20

        tensors = []
        size_change = 0
        for _ in range(N):
            tensors.append(torch.zeros(H + size_change, W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_add(tensors, 1)
        size_change = 0
        for t in res: 
            self.assertEqual(t, torch.ones(H + size_change, W + size_change, device=device, dtype=dtype))
            size_change += 1

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list(self, device, dtype):
        tensors = []
        with self.assertRaisesRegex(RuntimeError, r"Tensor list can't be empty."):
            torch._foreach_add(tensors, 1)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        with self.assertRaisesRegex(RuntimeError, r"Only non overlapping and dense tensors are supported."):
            torch._foreach_add(tensors, 1)

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()

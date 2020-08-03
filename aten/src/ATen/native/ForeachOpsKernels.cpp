#include <ATen/ATen.h>
namespace at { namespace native {

std::vector<Tensor> foreach_add_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors.size(); i++) {
    auto temp = tensors[i].add(scalar);
    result.emplace_back(temp);
  }
  return result;
}

std::vector<Tensor> foreach_add_scalar__kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (int i = 0; i < tensors.size(); i++) {
    tensors[i].add_(scalar);
  }

  return tensors.vec();
}

}} // namespace at::native

#include <ATen/ATen.h>
namespace at { namespace native {

std::vector<Tensor> foreach_sub_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors.size(); i++) {
    auto temp = tensors[i].sub(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_sub_scalar__kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (int i = 0; i < tensors.size(); i++) {
    tensors[i].sub_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_div_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors.size(); i++) {
    auto temp = tensors[i].div(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_div_scalar__kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (int i = 0; i < tensors.size(); i++) {
    tensors[i].div_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_mul_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors.size(); i++) {
    auto temp = tensors[i].mul(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_mul_scalar__kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (int i = 0; i < tensors.size(); i++) {
    tensors[i].mul_(scalar);
  }

  return tensors.vec();
}

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

std::vector<Tensor> foreach_add_list_kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].add(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_add_list__kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_sub_list_kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].sub(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_sub_list__kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].sub_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_mul_list_kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].mul(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_mul_list__kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].mul_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_div_list_kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].div(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_div_list__kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].div_(tensors2[i]);
  }

  return tensors1.vec();
}

}} // namespace at::native

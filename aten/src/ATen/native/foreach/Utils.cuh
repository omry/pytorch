#pragma once
#define ILP 4
#define CHUNK_SIZE 65536
#define BLOCK_SIZE 1024
namespace {

// TensorListMetadata has to be < 4KB - the limit for kernel launch argument
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

template<int n> struct TensorListMetadata
{
  void* addresses[n][depth_to_max_tensors[n-1]];
  int sizes[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  int block_to_chunk[depth_to_max_blocks[n-1]];
};

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

} // namespace

// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#if defined(MLX_CUDA_SM90A_ENABLED)

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

using namespace cute;

template <
    typename TileShapeMN = Shape<_128, _16>,
    typename ClusterShape = Shape<_1, _1, _1>,
    typename Element,
    typename Quant,
    typename GroupSize,
    typename F>
void qmm_sm90(
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* D,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    GroupSize group_size,
    F&& launch_kernel) {
  constexpr int kAlignmentA = 128 / sizeof_bits<Element>::value;
  constexpr int kAlignmentB = 128 / sizeof_bits<Quant>::value;
  constexpr int kTileShapeK =
      std::max(64, 128 * 8 / sizeof_bits<Element>::value);
  static_assert(group_size % kTileShapeK == 0);

  using Arch = cutlass::arch::Sm90;
  using Accumulator = float;
  using TileShape = decltype(append(TileShapeMN{}, Int<kTileShapeK>{}));

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator,
      Accumulator,
      // ElementC:
      void,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      // ElementD:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  // Note that A/B are swapped and transposed to use TMA epilogue.
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      // ElementA:
      tuple<Quant, Element, Element>,
      cutlass::layout::RowMajor,
      kAlignmentB,
      // ElementB:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      Accumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto dA = make_stride(k, Int<1>{}, m * k);
  auto dB = make_stride(k, Int<1>{}, n * k);
  auto dS = make_stride(Int<1>{}, n, n * k / group_size);
  auto dD = make_stride(Int<1>{}, n, m * n);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(n), int(m), int(k), int(l)},
      {B, dB, A, dA, S, dS, group_size, Z},
      {{1.f, 0.f}, D, dD, D, dD}};

  CHECK_CUTLASS_ERROR(gemm.can_implement(args));
  CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));

  auto* kernel = &cutlass::device_kernel<GemmKernel>;
  void* kernel_params[] = {const_cast<Gemm::Params*>(&gemm.params())};
  auto cluster = ClusterShape{};
  launch_kernel(
      reinterpret_cast<void*>(kernel),
      gemm.get_grid_shape(gemm.params()),
      GemmKernel::get_block_shape(),
      {static_cast<unsigned>(get<0>(cluster)),
       static_cast<unsigned>(get<1>(cluster)),
       static_cast<unsigned>(get<2>(cluster))},
      GemmKernel::SharedStorageSize,
      kernel_params);
}

template <
    typename TileShapeMN = Shape<_128, _16>,
    typename ClusterShape = Shape<_1, _1, _1>,
    typename Element,
    typename Quant,
    typename GroupSize,
    typename F>
void gather_qmm_sm90(
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* D,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    GroupSize group_size,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    const Element** A_ptrs,
    const Quant** B_ptrs,
    const Element** S_ptrs,
    const Element** Z_ptrs,
    Element** D_ptrs,
    F&& launch_kernel) {
  constexpr int kAlignmentA = 128 / sizeof_bits<Element>::value;
  constexpr int kAlignmentB = 128 / sizeof_bits<Quant>::value;
  constexpr int kTileShapeK =
      std::max(64, 128 * 8 / sizeof_bits<Element>::value);
  static_assert(group_size % kTileShapeK == 0);

  using Arch = cutlass::arch::Sm90;
  using Accumulator = float;
  using TileShape = decltype(append(TileShapeMN{}, Int<kTileShapeK>{}));

  // Use PtrArray schedules instead of strided batching.
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator,
      Accumulator,
      void,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      EpilogueSchedule>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      tuple<Quant, Element, Element>,
      cutlass::layout::RowMajor,
      kAlignmentB,
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      Accumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int, int, int, int>>,
      Mainloop,
      Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideS = typename Mainloop::NonVoidStrideScale;

  StrideA dA;
  get<0>(dA) = k; // dM = K
  StrideB dB;
  get<0>(dB) = k; // dN = K
  StrideD dD;
  get<1>(dD) = n; // dN = N
  StrideS dS;
  get<1>(dS) = n; // dN = N

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kArray,
      {{int(n), int(m), int(k), int(l)}},
      {B_ptrs, dB, A_ptrs, dA, S_ptrs, &dS, group_size, Z_ptrs},
      {{1.f, 0.f}, nullptr, dD, D_ptrs, dD}};

  CHECK_CUTLASS_ERROR(gemm.can_implement(args));
  CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));

  auto* kernel = &cutlass::device_kernel<GemmKernel>;
  void* kernel_params[] = {const_cast<Gemm::Params*>(&gemm.params())};
  auto cluster = ClusterShape{};
  launch_kernel(
      reinterpret_cast<void*>(kernel),
      gemm.get_grid_shape(gemm.params()),
      GemmKernel::get_block_shape(),
      {static_cast<unsigned>(get<0>(cluster)),
       static_cast<unsigned>(get<1>(cluster)),
       static_cast<unsigned>(get<2>(cluster))},
      GemmKernel::SharedStorageSize,
      kernel_params);
}

} // namespace cutlass_gemm

namespace mlx::core {

inline array transpose_last_2_dims(
    const array& x,
    cu::CommandEncoder& encoder,
    const Stream& s) {
  array transposed = swapaxes_in_eval(x, -1, -2);
  array transposed_copy = contiguous_copy_gpu(transposed, s);
  encoder.add_temporary(transposed_copy);
  return transposed_copy;
}

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void dispatch_quant_types(int bits, const char* tag, F&& f) {
  if (bits == 2) {
    f.template operator()<cutlass::uint2b_t>();
  } else if (bits == 4) {
    f.template operator()<cutlass::uint4b_t>();
  } else if (bits == 8) {
    f.template operator()<uint8_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} {}-bit quantization is not supported.", tag, bits));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 64) {
    f(cute::Int<64>{});
  } else if (group_size == 128) {
    f(cute::Int<128>{});
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

template <typename TileShapeMN, typename ClusterShape>
void qmm_impl_sm90(
    const array& x,
    const array& w,
    const array& scales_,
    const array& biases_,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  const char* tag = "[quantized_matmul]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);

  // FIXME: Copy happens for every call.
  array scales = transpose_last_2_dims(scales_, encoder, s);
  array biases = transpose_last_2_dims(biases_, encoder, s);

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types(bits, tag, [&]<typename Quant>() {
      dispatch_groups(group_size, tag, [&](auto group_size) {
        encoder.set_input_array(x);
        encoder.set_input_array(w);
        encoder.set_input_array(scales);
        encoder.set_input_array(biases);
        encoder.set_output_array(out);
        cutlass_gemm::qmm_sm90(
            gpu_ptr<Element>(x),
            gpu_ptr<Quant>(w),
            gpu_ptr<Element>(scales),
            gpu_ptr<Element>(biases),
            gpu_ptr<Element>(out),
            m,
            n,
            k,
            l,
            group_size,
            [&](auto* kernel,
                dim3 num_blocks,
                dim3 block_dims,
                dim3 cluster_shape,
                uint32_t smem_bytes,
                void** args) {
              encoder.add_kernel_node_raw(
                  kernel,
                  num_blocks,
                  block_dims,
                  cluster_shape,
                  smem_bytes,
                  args);
            });
      });
    });
  });
}

template <typename A_t, typename B_t, typename S_t>
__global__ void build_gather_ptrs(
    const A_t* A,
    const B_t* B,
    const S_t* S,
    const S_t* Z,
    S_t* D,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    const A_t** A_ptrs,
    const B_t** B_ptrs,
    const S_t** S_ptrs,
    const S_t** Z_ptrs,
    S_t** D_ptrs,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t s_batch_stride,
    int64_t d_batch_stride,
    int num_batches) {
  int b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= num_batches) {
    return;
  }
  uint32_t a_idx = lhs_indices[b];
  uint32_t b_idx = rhs_indices[b];
  A_ptrs[b] = A + a_idx * a_batch_stride;
  B_ptrs[b] = B + b_idx * b_batch_stride;
  S_ptrs[b] = S + b_idx * s_batch_stride;
  Z_ptrs[b] = Z + b_idx * s_batch_stride;
  D_ptrs[b] = D + b * d_batch_stride;
}

template <typename TileShapeMN, typename ClusterShape>
void gather_qmm_impl_sm90(
    const array& x,
    const array& w,
    const array& scales_,
    const array& biases_,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  const char* tag = "[gather_qmm_sm90]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);

  // FIXME: Copy happens for every call.
  array scales = transpose_last_2_dims(scales_, encoder, s);
  array biases = transpose_last_2_dims(biases_, encoder, s);

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types(bits, tag, [&]<typename Quant>() {
      dispatch_groups(group_size, tag, [&](auto group_size) {
        encoder.set_input_array(x);
        encoder.set_input_array(w);
        encoder.set_input_array(scales);
        encoder.set_input_array(biases);
        encoder.set_input_array(lhs_indices);
        encoder.set_input_array(rhs_indices);
        encoder.set_output_array(out);

        // Allocate pointer arrays on GPU.
        size_t ptrs_bytes = l * sizeof(void*);
        auto A_ptrs_buf = cu::malloc_async(ptrs_bytes, encoder);
        auto B_ptrs_buf = cu::malloc_async(ptrs_bytes, encoder);
        auto S_ptrs_buf = cu::malloc_async(ptrs_bytes, encoder);
        auto Z_ptrs_buf = cu::malloc_async(ptrs_bytes, encoder);
        auto D_ptrs_buf = cu::malloc_async(ptrs_bytes, encoder);

        auto* A_ptrs = static_cast<const Element**>(A_ptrs_buf.raw_ptr());
        auto* B_ptrs = static_cast<const Quant**>(B_ptrs_buf.raw_ptr());
        auto* S_ptrs = static_cast<const Element**>(S_ptrs_buf.raw_ptr());
        auto* Z_ptrs = static_cast<const Element**>(Z_ptrs_buf.raw_ptr());
        auto* D_ptrs = static_cast<Element**>(D_ptrs_buf.raw_ptr());

        int64_t a_batch_stride = int64_t(m) * k;
        int64_t b_batch_stride = int64_t(n) * k;
        int64_t s_batch_stride =
            int64_t(n) * k / group_size; // scales are transposed
        int64_t d_batch_stride = int64_t(m) * n;

        // Build pointer arrays.
        int threads = std::min(l, 256);
        int blocks = (l + threads - 1) / threads;
        encoder.add_kernel_node(
            build_gather_ptrs<Element, Quant, Element>,
            dim3(blocks),
            dim3(threads),
            gpu_ptr<Element>(x),
            gpu_ptr<Quant>(w),
            gpu_ptr<Element>(scales),
            gpu_ptr<Element>(biases),
            gpu_ptr<Element>(out),
            gpu_ptr<uint32_t>(lhs_indices),
            gpu_ptr<uint32_t>(rhs_indices),
            A_ptrs,
            B_ptrs,
            S_ptrs,
            Z_ptrs,
            D_ptrs,
            a_batch_stride,
            b_batch_stride,
            s_batch_stride,
            d_batch_stride,
            l);

        // Launch gather GEMM.
        cutlass_gemm::gather_qmm_sm90(
            gpu_ptr<Element>(x),
            gpu_ptr<Quant>(w),
            gpu_ptr<Element>(scales),
            gpu_ptr<Element>(biases),
            gpu_ptr<Element>(out),
            m,
            n,
            k,
            l,
            group_size,
            gpu_ptr<uint32_t>(lhs_indices),
            gpu_ptr<uint32_t>(rhs_indices),
            A_ptrs,
            B_ptrs,
            S_ptrs,
            Z_ptrs,
            D_ptrs,
            [&](auto* kernel,
                dim3 num_blocks,
                dim3 block_dims,
                dim3 cluster_shape,
                uint32_t smem_bytes,
                void** args) {
              encoder.add_kernel_node_raw(
                  kernel,
                  num_blocks,
                  block_dims,
                  cluster_shape,
                  smem_bytes,
                  args);
            });

        // Keep pointer array buffers alive until graph execution completes.
        encoder.add_completed_handler(
            [A_ptrs_buf, B_ptrs_buf, S_ptrs_buf, Z_ptrs_buf, D_ptrs_buf]() {});
      });
    });
  });
}

} // namespace mlx::core

#define QMM_SM90_GPU(TileShapeMN, ClusterShape)                  \
  namespace mlx::core {                                          \
  template void qmm_impl_sm90<TileShapeMN, ClusterShape>(        \
      const array& x,                                            \
      const array& w,                                            \
      const array& scales,                                       \
      const array& biases,                                       \
      array& out,                                                \
      int bits,                                                  \
      int group_size,                                            \
      cu::CommandEncoder& encoder,                               \
      Stream s);                                                 \
  template void gather_qmm_impl_sm90<TileShapeMN, ClusterShape>( \
      const array& x,                                            \
      const array& w,                                            \
      const array& scales,                                       \
      const array& biases,                                       \
      const array& lhs_indices,                                  \
      const array& rhs_indices,                                  \
      array& out,                                                \
      int bits,                                                  \
      int group_size,                                            \
      cu::CommandEncoder& encoder,                               \
      Stream s);                                                 \
  }

#else

#define QMM_SM90_GPU(TileShapeMN, ClusterShape)

#endif // defined(MLX_CUDA_SM90A_ENABLED)

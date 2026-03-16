// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce_ops.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/scan.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename Op, typename T>
struct ScanResult {
  using type = T;
};

template <>
struct ScanResult<Sum, bool> {
  using type = int32_t;
};

template <typename T>
struct ReduceInit<LogAddExp, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::min();
  }
};

template <bool reverse, typename T, typename U, int N_READS>
inline __device__ void
load_values(int index, const T* in, U (&values)[N_READS], int size, U init) {
  int remaining = size - index * N_READS;
  if constexpr (reverse) {
    in += remaining - N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] =
            (N_READS - i - 1 < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        values[N_READS - i - 1] = cast_to<U>(in[i]);
      }
    }
  } else {
    in += index * N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        values[i] = (i < remaining) ? cast_to<U>(in[i]) : init;
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        values[i] = cast_to<U>(in[i]);
      }
    }
  }
}

template <bool reverse, int offset, typename T, int N_READS>
inline __device__ void
store_values(int index, T* out, T (&values)[N_READS], int size) {
  int start = index * N_READS + offset;
  int remaining = size - start;
  if constexpr (reverse) {
    out += remaining - N_READS;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        if (N_READS - i - 1 < remaining) {
          out[i] = values[N_READS - i - 1];
        }
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    out += start;
    if (remaining < N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        if (i < remaining) {
          out[i] = values[i];
        }
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        out[i] = values[i];
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
__global__ void contiguous_scan(const T* in, U* out, int32_t axis_size) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  in += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  __shared__ U warp_sums[WARP_SIZE];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U prefix = init;

  // Scan per block.
  for (int r = 0; r < cuda::ceil_div(axis_size, block.size() * N_READS); ++r) {
    int32_t index = r * block.size() + block.thread_rank();
    U values[N_READS];
    load_values<reverse>(index, in, values, axis_size, init);

    // Compute an inclusive scan per thread.
    for (int i = 1; i < N_READS; ++i) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums.
    U prev_thread_sum = cg::exclusive_scan(warp, values[N_READS - 1], op);
    if (warp.thread_rank() == 0) {
      prev_thread_sum = init;
    }

    // Write wrap's sum to shared memory.
    if (warp.thread_rank() == WARP_SIZE - 1) {
      warp_sums[warp.meta_group_rank()] =
          op(prev_thread_sum, values[N_READS - 1]);
    }
    block.sync();

    // Compute exclusive scan of warp sums.
    if (warp.meta_group_rank() == 0) {
      U prev_warp_sum =
          cg::exclusive_scan(warp, warp_sums[warp.thread_rank()], op);
      if (warp.thread_rank() == 0) {
        prev_warp_sum = init;
      }
      warp_sums[warp.thread_rank()] = prev_warp_sum;
    }
    block.sync();

    // Compute the output.
    for (int i = 0; i < N_READS; ++i) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], warp_sums[warp.meta_group_rank()]);
      values[i] = op(values[i], prev_thread_sum);
    }

    // Write the values.
    if (inclusive) {
      store_values<reverse, 0>(index, out, values, axis_size);
    } else {
      store_values<reverse, 1>(index, out, values, axis_size);
      if (reverse) {
        if (block.thread_rank() == 0 && index == 0) {
          out[axis_size - 1] = init;
        }
      } else {
        if (block.thread_rank() == 0 && index == 0) {
          out[0] = init;
        }
      }
    }
    block.sync();

    // Share the prefix.
    if ((warp.meta_group_rank() == warp.meta_group_size() - 1) &&
        (warp.thread_rank() == WARP_SIZE - 1)) {
      warp_sums[0] = values[N_READS - 1];
    }
    block.sync();
    prefix = warp_sums[0];
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    int BM,
    int BN,
    bool inclusive,
    bool reverse>
__global__ void strided_scan(
    const T* in,
    U* out,
    int32_t axis_size,
    int64_t stride,
    int64_t stride_blocks) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int BN_pad = WARP_SIZE + 16 / sizeof(U);
  constexpr int n_warps = BN / N_READS;
  constexpr int n_scans = BN / n_warps;

  __shared__ U read_buffer[BM * BN_pad];

  Op op;
  U init = ReduceInit<Op, T>::value();
  U values[n_scans];
  U prefix[n_scans];
  for (int i = 0; i < n_scans; ++i) {
    prefix[i] = init;
  }

  // Compute offsets.
  int64_t offset = (grid.block_rank() / stride_blocks) * axis_size * stride;
  int64_t global_index_x = (grid.block_rank() % stride_blocks) * BN;
  uint32_t read_offset_y = (block.thread_rank() * N_READS) / BN;
  uint32_t read_offset_x = (block.thread_rank() * N_READS) % BN;
  uint32_t scan_offset_y = warp.thread_rank();
  uint32_t scan_offset_x = warp.meta_group_rank() * n_scans;

  uint32_t stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  U* read_into = read_buffer + read_offset_y * BN_pad + read_offset_x;
  U* read_from = read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint32_t j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread.
    uint32_t index_y = j + read_offset_y;
    uint32_t check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }

    // Read in SM.
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; ++i) {
        read_into[i] = in[index_y * stride + i];
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = in[index_y * stride + i];
        } else {
          read_into[i] = init;
        }
      }
    }
    block.sync();

    // Read strided into registers.
    for (int i = 0; i < n_scans; ++i) {
      values[i] = read_from[i];
    }

    // Perform the scan.
    for (int i = 0; i < n_scans; ++i) {
      values[i] = cg::inclusive_scan(warp, values[i], op);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = warp.shfl(values[i], WARP_SIZE - 1);
    }

    // Write to SM.
    for (int i = 0; i < n_scans; ++i) {
      read_from[i] = values[i];
    }
    block.sync();

    // Write to device memory.
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
          for (int i = 0; i < N_READS; ++i) {
            out[index_y * stride + i] = init;
          }
        } else {
          for (int i = 0; i < N_READS; ++i) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; ++i) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}

template <typename T, typename U, typename Op, int N_READS, bool reverse>
__global__ void contiguous_scan_tile(
    const T* in,
    U* out,
    U* tile_aggs,
    int32_t axis_size,
    int32_t tile_size,
    int32_t num_tiles) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int scan_id = grid.block_rank() / num_tiles;
  int tile_id = grid.block_rank() % num_tiles;

  // Compute tile boundaries.
  int32_t tile_start, tile_len;
  if constexpr (reverse) {
    int32_t tile_end = axis_size - tile_id * tile_size;
    tile_start = (tile_end > tile_size) ? (tile_end - tile_size) : 0;
    tile_len = tile_end - tile_start;
  } else {
    tile_start = tile_id * tile_size;
    tile_len = min(tile_size, axis_size - tile_start);
  }

  const T* tile_in = in + scan_id * axis_size + tile_start;
  U* tile_out = out + scan_id * axis_size + tile_start;

  __shared__ U warp_sums[WARP_SIZE];

  Op op;
  U init = ReduceInit<Op, T>::value();

  // Load values for this tile.
  int32_t index = block.thread_rank();
  U values[N_READS];
  load_values<reverse>(index, tile_in, values, tile_len, init);

  // Inclusive scan per thread.
  for (int i = 1; i < N_READS; ++i) {
    values[i] = op(values[i], values[i - 1]);
  }

  // Exclusive scan of thread sums within warp.
  U prev_thread_sum = cg::exclusive_scan(warp, values[N_READS - 1], op);
  if (warp.thread_rank() == 0) {
    prev_thread_sum = init;
  }

  // Write warp sum to shared memory.
  if (warp.thread_rank() == WARP_SIZE - 1) {
    warp_sums[warp.meta_group_rank()] =
        op(prev_thread_sum, values[N_READS - 1]);
  }
  block.sync();

  // Exclusive scan of warp sums.
  if (warp.meta_group_rank() == 0) {
    U prev_warp_sum =
        cg::exclusive_scan(warp, warp_sums[warp.thread_rank()], op);
    if (warp.thread_rank() == 0) {
      prev_warp_sum = init;
    }
    warp_sums[warp.thread_rank()] = prev_warp_sum;
  }
  block.sync();

  // Compute inclusive scan output.
  for (int i = 0; i < N_READS; ++i) {
    values[i] = op(values[i], warp_sums[warp.meta_group_rank()]);
    values[i] = op(values[i], prev_thread_sum);
  }

  // Write inclusive scan values.
  store_values<reverse, 0>(index, tile_out, values, tile_len);

  // Write tile aggregate (last thread has the full aggregate).
  if (block.thread_rank() == block.size() - 1) {
    tile_aggs[scan_id * num_tiles + tile_id] = values[N_READS - 1];
  }
}

template <typename U, typename Op, int N_READS, bool inclusive, bool reverse>
__global__ void propagate_prefix(
    U* out,
    const U* tile_prefixes,
    int32_t axis_size,
    int32_t tile_size,
    int32_t num_tiles) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  int scan_id = grid.block_rank() / num_tiles;
  int tile_id = grid.block_rank() % num_tiles;

  U prefix = tile_prefixes[scan_id * num_tiles + tile_id];

  Op op;
  U init = ReduceInit<Op, U>::value();

  // Compute tile boundaries.
  int32_t tile_start, tile_len;
  if constexpr (reverse) {
    int32_t tile_end = axis_size - tile_id * tile_size;
    tile_start = (tile_end > tile_size) ? (tile_end - tile_size) : 0;
    tile_len = tile_end - tile_start;
  } else {
    tile_start = tile_id * tile_size;
    tile_len = min(tile_size, axis_size - tile_start);
  }

  U* tile_out = out + scan_id * axis_size + tile_start;
  int idx = block.thread_rank() * N_READS;

  if constexpr (inclusive) {
    // Simply add prefix to each element.
    if (idx + N_READS <= tile_len) {
      for (int i = 0; i < N_READS; ++i) {
        tile_out[idx + i] = op(tile_out[idx + i], prefix);
      }
    } else {
      for (int i = 0; i < N_READS; ++i) {
        if (idx + i < tile_len) {
          tile_out[idx + i] = op(tile_out[idx + i], prefix);
        }
      }
    }
  } else {
    // Exclusive: read inclusive values, shift by 1, then add prefix.
    __shared__ U warp_boundary[WARP_SIZE];

    // Read inclusive scan values from Phase 1.
    U values[N_READS];
    for (int i = 0; i < N_READS; ++i) {
      values[i] = (idx + i < tile_len) ? tile_out[idx + i] : init;
    }

    if constexpr (!reverse) {
      // Forward exclusive: shift right.
      U my_last = values[N_READS - 1];
      U prev = warp.shfl_up(my_last, 1);

      if (warp.thread_rank() == WARP_SIZE - 1) {
        warp_boundary[warp.meta_group_rank()] = my_last;
      }
      block.sync();

      if (warp.thread_rank() == 0) {
        if (warp.meta_group_rank() == 0) {
          prev = init;
        } else {
          prev = warp_boundary[warp.meta_group_rank() - 1];
        }
      }

      for (int i = N_READS - 1; i > 0; --i) {
        values[i] = values[i - 1];
      }
      values[0] = prev;
    } else {
      // Reverse exclusive: shift left.
      U my_first = values[0];
      U next = warp.shfl_down(my_first, 1);

      if (warp.thread_rank() == 0) {
        warp_boundary[warp.meta_group_rank()] = my_first;
      }
      block.sync();

      if (warp.thread_rank() == WARP_SIZE - 1) {
        int num_warps = block.size() / WARP_SIZE;
        if (warp.meta_group_rank() == num_warps - 1) {
          next = init;
        } else {
          next = warp_boundary[warp.meta_group_rank() + 1];
        }
      }

      for (int i = 0; i < N_READS - 1; ++i) {
        values[i] = values[i + 1];
      }
      values[N_READS - 1] = next;
    }

    // Add prefix to all values.
    for (int i = 0; i < N_READS; ++i) {
      values[i] = op(values[i], prefix);
    }

    // Write back.
    for (int i = 0; i < N_READS; ++i) {
      if (idx + i < tile_len) {
        tile_out[idx + i] = values[i];
      }
    }
  }
}

} // namespace cu

template <typename F>
void dispatch_scan_ops(Scan::ReduceType scan_op, F&& f) {
  if (scan_op == Scan::ReduceType::Max) {
    f(type_identity<cu::Max>{});
  } else if (scan_op == Scan::ReduceType::Min) {
    f(type_identity<cu::Min>{});
  } else if (scan_op == Scan::ReduceType::Sum) {
    f(type_identity<cu::Sum>{});
  } else if (scan_op == Scan::ReduceType::Prod) {
    f(type_identity<cu::Prod>{});
  } else if (scan_op == Scan::ReduceType::LogAddExp) {
    f(type_identity<cu::LogAddExp>{});
  } else {
    throw std::invalid_argument("Unknown reduce type.");
  }
}

template <typename Op>
const char* op_to_string() {
  if (cuda::std::is_same_v<Op, cu::Max>) {
    return "Max";
  } else if (cuda::std::is_same_v<Op, cu::Min>) {
    return "Min";
  } else if (cuda::std::is_same_v<Op, cu::Sum>) {
    return "Sum";
  } else if (cuda::std::is_same_v<Op, cu::Prod>) {
    return "Prod";
  } else if (cuda::std::is_same_v<Op, cu::LogAddExp>) {
    return "LogAddExp";
  } else {
    throw std::invalid_argument("Unknown op.");
  }
}

template <typename Op, typename T>
constexpr bool supports_scan_op() {
  if constexpr (cuda::std::is_same_v<Op, LogAddExp>) {
    return is_inexact_v<T>;
  } else {
    return true;
  }
}

void scan_gpu_inplace(
    array in,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s) {
  auto& encoder = cu::get_command_encoder(s);
  constexpr int N_READS = 4;
  int32_t axis_size = in.shape(axis);
  bool contiguous = in.strides()[axis] == 1;

  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_scan_ops(reduce_type, [&](auto scan_op_tag) {
      using Op = MLX_GET_TYPE(scan_op_tag);
      if constexpr (supports_scan_op<Op, T>()) {
        using U = typename cu::ScanResult<Op, T>::type;
        dispatch_bool(inclusive, [&](auto inclusive_tag) {
          dispatch_bool(reverse, [&](auto reverse_tag) {
            if (contiguous) {
              int block_dim = cuda::ceil_div(axis_size, N_READS);
              block_dim = cuda::ceil_div(block_dim, WARP_SIZE) * WARP_SIZE;
              block_dim = std::min(block_dim, WARP_SIZE * WARP_SIZE);
              int tile_size = block_dim * N_READS;

              int num_tiles = cuda::ceil_div(axis_size, tile_size);
              int num_scans = static_cast<int>(in.data_size() / axis_size);

              // Use multi-block only when there is a single scan instance with
              // a large axis.
              bool use_multi_block = num_scans == 1 && num_tiles >= 16;

              if (!use_multi_block) {
                auto kernel = cu::contiguous_scan<
                    T,
                    U,
                    Op,
                    N_READS,
                    inclusive_tag.value,
                    reverse_tag.value>;
                encoder.set_input_array(in);
                encoder.set_output_array(out);
                encoder.add_kernel_node(
                    kernel,
                    in.data_size() / axis_size,
                    block_dim,
                    gpu_ptr<T>(in),
                    gpu_ptr<U>(out),
                    axis_size);
              } else {
                // Multi-block path: split each scan into tiles processed by
                // separate blocks, then propagate inter-tile prefixes.

                // Allocate workspace for tile aggregates.
                size_t agg_bytes =
                    static_cast<size_t>(num_scans) * num_tiles * sizeof(U);
                array tile_aggs(
                    {num_scans * num_tiles}, out.dtype(), nullptr, {});
                tile_aggs.set_data(cu::malloc_async(agg_bytes, encoder));
                encoder.add_temporary(tile_aggs);

                // Inclusive scan within each tile.
                auto tile_kernel = cu::
                    contiguous_scan_tile<T, U, Op, N_READS, reverse_tag.value>;
                encoder.set_input_array(in);
                encoder.set_output_array(out);
                encoder.set_output_array(tile_aggs);
                encoder.add_kernel_node(
                    tile_kernel,
                    num_scans * num_tiles,
                    block_dim,
                    gpu_ptr<T>(in),
                    gpu_ptr<U>(out),
                    gpu_ptr<U>(tile_aggs),
                    axis_size,
                    tile_size,
                    num_tiles);

                // Exclusive forward scan of tile aggregates.
                auto agg_kernel = cu::contiguous_scan<
                    U,
                    U,
                    Op,
                    N_READS,
                    false /* exclusive */,
                    false /* forward */>;
                int agg_block_dim = cuda::ceil_div(num_tiles, N_READS);
                agg_block_dim =
                    cuda::ceil_div(agg_block_dim, WARP_SIZE) * WARP_SIZE;
                agg_block_dim = std::min(agg_block_dim, WARP_SIZE * WARP_SIZE);
                encoder.set_input_array(tile_aggs);
                encoder.set_output_array(tile_aggs);
                encoder.add_kernel_node(
                    agg_kernel,
                    num_scans,
                    agg_block_dim,
                    gpu_ptr<U>(tile_aggs),
                    gpu_ptr<U>(tile_aggs),
                    num_tiles);

                // Propagate prefix to each tile.
                auto prop_kernel = cu::propagate_prefix<
                    U,
                    Op,
                    N_READS,
                    inclusive_tag.value,
                    reverse_tag.value>;
                encoder.set_input_array(tile_aggs);
                encoder.set_input_array(out);
                encoder.set_output_array(out);
                encoder.add_kernel_node(
                    prop_kernel,
                    num_scans * num_tiles,
                    block_dim,
                    gpu_ptr<U>(out),
                    gpu_ptr<U>(tile_aggs),
                    axis_size,
                    tile_size,
                    num_tiles);
              }
            } else {
              constexpr int BM = WARP_SIZE;
              constexpr int BN = WARP_SIZE;
              auto kernel = cu::strided_scan<
                  T,
                  U,
                  Op,
                  N_READS,
                  BM,
                  BN,
                  inclusive_tag.value,
                  reverse_tag.value>;
              int64_t stride = in.strides()[axis];
              int64_t stride_blocks = cuda::ceil_div(stride, BN);
              dim3 num_blocks = get_2d_grid_dims(
                  in.shape(), in.strides(), axis_size * stride);
              if (num_blocks.x * stride_blocks <= UINT32_MAX) {
                num_blocks.x *= stride_blocks;
              } else {
                num_blocks.y *= stride_blocks;
              }
              int block_dim = (BN / N_READS) * WARP_SIZE;
              encoder.set_input_array(in);
              encoder.set_output_array(out);
              encoder.add_kernel_node(
                  kernel,
                  num_blocks,
                  block_dim,
                  gpu_ptr<T>(in),
                  gpu_ptr<U>(out),
                  axis_size,
                  stride,
                  stride_blocks);
            }
          });
        });
      } else {
        throw std::runtime_error(
            fmt::format(
                "Can not do scan op {} on inputs of {} with result of {}.",
                op_to_string<Op>(),
                dtype_to_string(in.dtype()),
                dtype_to_string(out.dtype())));
      }
    });
  });
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Scan::eval_gpu");
  assert(inputs.size() == 1);
  auto in = inputs[0];
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          cu::malloc_async(in.data_size() * out.itemsize(), encoder),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    in = contiguous_copy_gpu(in, s);
    out.copy_shared_buffer(in);
  }

  scan_gpu_inplace(in, out, reduce_type_, axis_, reverse_, inclusive_, s);
}

} // namespace mlx::core

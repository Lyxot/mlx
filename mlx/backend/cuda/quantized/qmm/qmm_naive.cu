// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/cute_dequant.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

// clang-format off

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

using namespace cute;

template <typename Element, typename SmemLayoutA, typename SmemLayoutB>
struct SharedStorage {
  ArrayEngine<Element, cosize_v<SmemLayoutA>> A;
  ArrayEngine<Element, cosize_v<SmemLayoutB>> B;
};

__device__ __forceinline__ void
cute_naive_dequant(auto w, auto s, auto z, auto out) {
  using Element = typename decltype(out)::value_type;
  using Quant = typename decltype(w)::value_type;
  using Scale = typename decltype(s)::value_type;
  transform(w, out, [](Quant q) { return Element(q); } );
  transform(out, s, out, [](Element e, Scale s) { return e * Element(s); });
  if constexpr (quant_has_bias_v<Quant>) {
    transform(out, z, out, plus{});
  }
}

__device__ __forceinline__ void
cute_dequant(auto w, auto s, auto z, auto out) {
  if constexpr (stride(coalesce(w.layout())) == Int<1>{} &&
                is_static_v<decltype(s.layout())>) {
    cute_vectorized_dequant(w, s, z, out);
  } else {
    cute_naive_dequant(w, s, z, out);
  }
}

template <bool HasKResidue,
          typename TKRes,
          typename TCopyA, typename TApA, typename TAcA,
          typename TAgA, typename TArA, typename TAsA,
          typename TCopyB, typename TBpB, typename TBcB,
          typename TBgB, typename TBrB, typename TBrBdq, typename TBsB,
          typename TBgS, typename TBrS, typename TBgZ, typename TBrZ,
          typename TMma, typename TCsA, typename TCsB, typename TCrC>
__device__ void qmm_gemm(
    TKRes k_residue,
    TCopyA& copy_a, TApA& tApA, TAcA& tAcA,
    TAgA& tAgA, TArA& tArA, TAsA& tAsA,
    TCopyB& copy_b, TBpB& tBpB, TBcB& tBcB,
    TBgB& tBgB, TBrB& tBrB, TBrBdq& tBrB_dq, TBsB& tBsB,
    TBgS& tBgS, TBrS& tBrS, TBgZ& tBgZ, TBrZ& tBrZ,
    TMma& mma, TCsA& tCsA, TCsB& tCsB, TCrC& tCrC) {
  // GMEM => RMEM.
  auto fetch_gmem = [&](int tile) {
    copy_if(copy_a, tApA, tAgA(_,_,_,tile), tArA);
    copy_if(copy_b, tBpB, tBgB(_,_,_,tile), tBrB);
    copy(tBgS(_,_,_,tile), tBrS);
    copy(tBgZ(_,_,_,tile), tBrZ);
  };
  // RMEM => SMEM.
  auto store_smem = [&]() {
    __syncthreads();
    copy(tArA, tAsA);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tBrB); ++n) {
        cute_dequant(tBrB(_,n,k), tBrS(_,n,k), tBrZ(_,n,k), tBrB_dq(_,n,k));
      }
    }
    copy(tBrB_dq, tBsB);
    __syncthreads();
  };

  // Clear the rmem tiles to account for predicated off loads.
  if constexpr (HasKResidue) {
    clear(tArA);
    clear(tBrB);
    clear(tBrS);
    clear(tBrZ);
  }

  // Prefetch first tile.
  if constexpr (HasKResidue) {
    Tensor tAgA_k = tAgA(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tArA); ++k) {
      if (get<1>(tAcA(0,0,k)) >= -k_residue) {
        copy_if(copy_a, tApA(_,k), tAgA_k(_,_,k), tArA(_,_,k));
      }
    }
    Tensor tBgB_k = tBgB(_,_,_,0);
    Tensor tBgS_k = tBgS(_,_,_,0);
    Tensor tBgZ_k = tBgZ(_,_,_,0);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tBrB); ++k) {
      if (get<1>(tBcB(0,0,k)) >= -k_residue) {
        copy_if(copy_b, tBpB(_,k), tBgB_k(_,_,k), tBrB(_,_,k));
        copy(tBgS_k(_,_,k), tBrS(_,_,k));
        copy(tBgZ_k(_,_,k), tBrZ(_,_,k));
      }
    }
  } else {
    fetch_gmem(0);
  }

  // Clear accumulators.
  clear(tCrC);

  // Loop over CTA tiles.
  auto K_TILE_MAX = size<3>(tAgA);
  for (int tile = 0; tile < K_TILE_MAX; ++tile) {
    store_smem();
    if constexpr (HasKResidue) {
      // Avoid fetching full 0th-tile when there is residue.
      if (K_TILE_MAX > 1) {
        fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
      }
    } else {
      fetch_gmem((tile + 1 < K_TILE_MAX) ? tile + 1 : tile);
    }
    gemm(mma, tCsA, tCsB, tCrC);
  }
}

template <bool HasKResidue, typename ProblemShape, typename CtaTiler,
          typename Element, typename Quant, typename Scale,
          typename StrideA, typename SmemLayoutA, typename TiledCopyA,
          typename StrideB, typename SmemLayoutB, typename TiledCopyB,
          typename StrideC, typename LayoutS, typename TiledMma>
__global__ void qmm_naive_kernel(
    ProblemShape shape_MNKL, CtaTiler cta_tiler,
    const Element* A, StrideA dA, SmemLayoutA sA_layout, TiledCopyA copy_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB copy_b,
          Element* C, StrideC dC,
    const Scale* S, const Element* Z, LayoutS S_layout,
    const uint32_t* lhs_indices, const uint32_t* rhs_indices,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));
  CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));

  int thread_idx = int(threadIdx.x);
  auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);

  auto m_max_coord = size<0>(shape_MNKL) - size<0>(cta_tiler) * m_coord; // M - BLK_M * m_coord
  auto n_max_coord = size<1>(shape_MNKL) - size<1>(cta_tiler) * n_coord; // N - BLK_N * n_coord

  // Shift tensor so we handle residue of K in the 0th tile.
  auto shape_K = size<2>(shape_MNKL);
  auto bK = size<2>(cta_tiler);
  auto k_residue = shape_K - bK * ceil_div(shape_K, bK);
  if constexpr (HasKResidue) {
    A += k_residue * get<1>(dA);
    B += k_residue * get<1>(dB) * cuda::std::min(8, sizeof_bits_v<Quant>) / 8;
    S += k_residue * stride<1>(S_layout);
    Z += k_residue * stride<1>(S_layout);
  }

  // Represent the full tensors.
  Tensor mA_mkl = make_tensor(make_gmem_ptr(A),        select<0,2,3>(shape_MNKL), dA); // (M,K,L)
  Tensor mB_nkl = make_tensor(make_gmem_ptr<Quant>(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
  Tensor mC_mnl = make_tensor(make_gmem_ptr(C),        select<0,1,3>(shape_MNKL), dC); // (M,N,L)

  Tensor mS_nkl = make_tensor(make_gmem_ptr(S), S_layout); // (N,(group_size,K/group_size),L)
  Tensor mZ_nkl = make_tensor(make_gmem_ptr(Z), S_layout); // (N,(group_size,K/group_size),L)

  // For gather, use index lookup for input batch slicing.
  uint32_t a_batch = lhs_indices ? lhs_indices[l_coord] : l_coord;
  uint32_t b_batch = rhs_indices ? rhs_indices[l_coord] : l_coord;

  // Get batch slice.
  Tensor mA = mA_mkl(_,_,a_batch); // (M,K)
  Tensor mB = mB_nkl(_,_,b_batch); // (N,K)
  Tensor mC = mC_mnl(_,_,l_coord); // (M,N)

  Tensor mS = mS_nkl(_,_,b_batch); // (N,(group_size,K/group_size))
  Tensor mZ = mZ_nkl(_,_,b_batch); // (N,(group_size,K/group_size))

  // Get the appropriate blocks for this thread block.
  auto cta_coord = make_coord(m_coord, n_coord, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<Element, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K)

  // Partition the copying of A/B/C tiles across the threads.
  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA); // (ACPY,ACPY_M,ACPY_K)
  Tensor tArA = make_fragment_like(tAsA);   // (ACPY,ACPY_M,ACPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBgB = thr_copy_b.partition_S(gB);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);        // (BCPY,BCPY_N,BCPY_K)
  Tensor tBrB = make_fragment_like<Quant>(tBsB);   // (BCPY,BCPY_M,BCPY_K)
  Tensor tBrB_dq = make_fragment_like(tBsB);       // (BCPY,BCPY_M,BCPY_K)
  Tensor tBgS = thr_copy_b.partition_S(gS);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrS = make_fragment_like(tBgS(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)
  Tensor tBgZ = thr_copy_b.partition_S(gZ);        // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBrZ = make_fragment_like(tBgZ(_,_,_,0)); // (BCPY,BCPY_N,BCPY_K)

  // MMA.
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA);       // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);       // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);       // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  // Predicates for m/n bounds.
  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{}); // (CPY_M,CPY_K)
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{}); // (CPY_N,CPY_K)
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K)
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (M,N)
  Tensor tAcA = thr_copy_a.partition_S(cA); // (CPY,CPY_M,CPY_K)
  Tensor tBcB = thr_copy_b.partition_S(cB); // (CPY,CPY_N,CPY_K)
  Tensor tCcC = thr_mma.partition_C(cC);    // (MMA,MMA_M,MMA_N)
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }

  qmm_gemm<HasKResidue>(
      k_residue,
      copy_a, tApA, tAcA, tAgA, tArA, tAsA,
      copy_b, tBpB, tBcB, tBgB, tBrB, tBrB_dq, tBsB,
      tBgS, tBrS, tBgZ, tBrZ,
      mma, tCsA, tCsB, tCrC);

  // Epilogue.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if ((get<0>(tCcC(i)) < m_max_coord) && (get<1>(tCcC(i)) < n_max_coord)) {
      tCgC(i) = Element(tCrC(i));
    }
  }
}

template <bool KMajor>
inline constexpr auto make_matrix_stride(auto m, auto k) {
  if constexpr (KMajor) {
    return cute::make_stride(k, cute::Int<1>{}, m * k);
  } else {
    return cute::make_stride(cute::Int<1>{}, m, m * k);
  }
}

template <bool KMajor = true>
inline constexpr auto make_smem_layout(auto bM, auto bK) {
  // TODO: Calculate swizzle based on tile shape.
  if constexpr (KMajor) {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape <_8,Shape <_8, _8>>,
                                      Stride<_8,Stride<_1,_64>>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  } else {
    auto swizzle = composition(Swizzle<3,3,3>{},
                               Layout<Shape<_64,_1>, Stride<_1,_64>>{});
    return tile_to_shape(swizzle, make_shape(bM, bK));
  }
}

template <int TileM, bool SM80, typename Element>
inline constexpr auto make_tiled_mma() {
  using Atom = std::conditional_t<
      SM80,
      std::conditional_t<
          std::is_same_v<Element, half_t>,
          SM80_16x8x16_F32F16F16F32_TN,
          std::conditional_t<
              std::is_same_v<Element, bfloat16_t>,
              SM80_16x8x16_F32BF16BF16F32_TN,
              UniversalFMA<float>
          >
      >,
      UniversalFMA<float, Element, Element>>;
  if constexpr (!SM80 || std::is_same_v<Element, float>) {
    return make_tiled_mma(Atom{}, Layout<Shape<_16,_8,_1>>{});
  } else {
    if constexpr (TileM >= 32) {
      return make_tiled_mma(Atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<_32,_32,_16>{});
    } else {
      return make_tiled_mma(Atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<_16,_32,_16>{});
    }
  }
}

template <typename T, bool KMajor = true, bool HasKResidue = false>
inline auto make_tiled_copy(auto num_threads, auto bM, auto bK) {
  // TODO: Only do 1-element read for the tile of residue.
  auto n_read = Int<HasKResidue ? 1 : 8>{};
  auto atom = Copy_Atom<UniversalCopy<uint_bit_t<n_read * sizeof_bits_v<T>>>, T>{};
  if constexpr (KMajor) {
    auto k_threads = bK / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(Int<num_threads / k_threads>{}, k_threads), LayoutRight{}),
        make_layout(make_shape(Int<1>{}, n_read)));
  } else {
    auto m_threads = bM / n_read;
    return make_tiled_copy(
        atom,
        make_layout(make_shape(m_threads, Int<num_threads / m_threads>{}), LayoutLeft{}),
        make_layout(make_shape(n_read, Int<1>{})));
  }
}

template <bool KMajor>
inline constexpr auto make_scales_layout(auto n, auto k, auto l, auto group_size) {
  if constexpr (KMajor) {
    return make_layout(
        make_shape(n, make_shape(group_size, k / group_size), l),
        make_stride(k / group_size, Stride<_0,_1>{}, n * k / group_size));
  } else {
    return make_layout(
        make_shape(make_shape(group_size, n / group_size), k, l),
        make_stride(Stride<_0,_1>{}, n / group_size, n * k / group_size));
  }
}

template <int TileM = 16, bool KMajor = true, bool HasKResidue = false, bool SM80 = true,
          typename Element, typename Quant, typename Scale>
void qmm_naive(
    const Element* A,
    const Quant*   B,
    const Scale*   S,
    const Element* Z,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int l,
    bool broadcast_b,
    auto group_size,
    auto&& launch_kernel) {
  // Define shapes (dynamic).
  auto prob_shape = make_shape(m, n, k, l); // (M,N,K,L)

  // Define TN strides (mixed).
  auto dA = make_stride(k, Int<1>{}, m * k);  // (dM,dK,dL)
  auto dB = make_matrix_stride<KMajor>(n, k); // (dN,dK,dL)
  auto dC = make_stride(n, Int<1>{}, m * n);  // (dM,dN,dL)

  // Define layout of scales/biases (mixed).
  auto S_layout = make_scales_layout<KMajor>(n, k, l, group_size);

  // Handle broadcasting.
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(stride(S_layout)) = 0;
  }

  // Define CTA tile sizes (static).
  auto bM = Int<TileM>{};
  auto bN = Int<(!SM80 && group_size > 64) ? 64 : 128>{};
  auto bK = Int<max(64, group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, SM80, Element>();
  auto num_threads = size(mma);

  // Define the A/B smem layouts (static).
  auto sA_layout = make_smem_layout(bM, bK);
  auto sB_layout = make_smem_layout<KMajor>(bN, bK);

  // Atoms.
  TiledCopy copy_a = make_tiled_copy<Element, true, HasKResidue>(num_threads, bM, bK);
  TiledCopy copy_b = make_tiled_copy<Quant, KMajor>(num_threads, bN, bK);

  auto* kernel = &qmm_naive_kernel<
      HasKResidue, decltype(prob_shape), decltype(cta_tiler),
      Element, Quant, Scale,
      decltype(dA), decltype(sA_layout), decltype(copy_a),
      decltype(dB), decltype(sB_layout), decltype(copy_b),
      decltype(dC), decltype(S_layout), decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), l);
  dim3 block_dims(num_threads);
  void* args[] = {
      &prob_shape, &cta_tiler,
      &A, &dA, &sA_layout, &copy_a,
      &B, &dB, &sB_layout, &copy_b,
      &C, &dC,
      &S, &Z, &S_layout,
      &lhs_indices, &rhs_indices,
      &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
}

template <bool HasKResidue, typename CtaTiler,
          typename Element, typename Quant, typename Scale,
          typename StrideB, typename SmemLayoutA, typename TiledCopyA,
          typename SmemLayoutB, typename TiledCopyB,
          typename LayoutS, typename TiledMma>
__global__ void qmm_sorted_naive_kernel(
    int M, int N, int K, int E,
    CtaTiler cta_tiler,
    const Element* A, SmemLayoutA sA_layout, TiledCopyA copy_a,
    const Quant*   B, StrideB dB, SmemLayoutB sB_layout, TiledCopyB copy_b,
          Element* C,
    const Scale* S, const Element* Z, LayoutS S_layout,
    const uint32_t* rhs_indices,
    TiledMma mma) {
  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));

  int thread_idx = int(threadIdx.x);
  int m_coord = int(blockIdx.x);
  int n_coord = int(blockIdx.y);

  int row_start = int(size<0>(cta_tiler)) * m_coord;
  int tile_m    = int(size<0>(cta_tiler));
  int tgp_m     = M - row_start;
  tgp_m = tgp_m < tile_m ? tgp_m : tile_m;

  auto m_max_coord = tgp_m;
  auto n_max_coord = N - int(size<1>(cta_tiler)) * n_coord;

  auto bK = size<2>(cta_tiler);
  auto k_residue = K - bK * ceil_div(K, bK);
  if constexpr (HasKResidue) {
    A += k_residue;  // stride in K dim = 1
    B += k_residue * get<1>(dB) * cuda::std::min(8, sizeof_bits_v<Quant>) / 8;
    S += k_residue * stride<1>(S_layout);
    if constexpr (quant_has_bias_v<Quant>) {
      Z += k_residue * stride<1>(S_layout);
    }
  }

  // Flat A (M, K) and C (M, N), pre-offset to this block's row range.
  Tensor mA = make_tensor(make_gmem_ptr(A + row_start * K),
                          make_shape(tgp_m, K),
                          make_stride(K, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C + row_start * N),
                          make_shape(tgp_m, N),
                          make_stride(N, Int<1>{}));
  // B: (N, K, E) indexed by expert.
  Tensor mB_nke = make_tensor(make_gmem_ptr<Quant>(B),
                              make_shape(N, K, E), dB);
  Tensor mS_nke = make_tensor(make_gmem_ptr(S), S_layout);
  Tensor mZ_nke = make_tensor(make_gmem_ptr(Z), S_layout);

  // m starts at 0 within the already-offset A/C tensors.
  auto cta_coord = make_coord(0, n_coord, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  // Shared memory buffers.
  extern __shared__ char shared_memory[];
  using SmemStorage = SharedStorage<Element, SmemLayoutA, SmemLayoutB>;
  SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
  Tensor tAgA = thr_copy_a.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);
  Tensor tArA = make_fragment_like(tAsA);

  ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
  Tensor tBsB = thr_copy_b.partition_D(sB);

  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
  Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
  Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC)));
  Tensor tAcA = thr_copy_a.partition_S(cA);
  Tensor tBcB = thr_copy_b.partition_S(cB);
  Tensor tCcC = thr_mma.partition_C(cC);
  CUTE_UNROLL
  for (int m = 0; m < size<0>(tApA); ++m) {
    tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
  }
  CUTE_UNROLL
  for (int n = 0; n < size<0>(tBpB); ++n) {
    tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
  }

  // Iterate over consecutive runs of equal rhs_indices within this m-tile.
  int offset = 0;
  while (offset < tgp_m) {
    uint32_t b = rhs_indices[row_start + offset];
    int offset_next = offset + 1;
    while (offset_next < tgp_m &&
           rhs_indices[row_start + offset_next] == b) {
      offset_next++;
    }

    Tensor mB = mB_nke(_,_,b);
    Tensor mS = mS_nke(_,_,b);
    Tensor mZ = mZ_nke(_,_,b);

    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
    Tensor gS = local_tile(mS, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gZ = local_tile(mZ, cta_tiler, cta_coord, Step< X,_1,_1>{});

    Tensor tBgB    = thr_copy_b.partition_S(gB);
    Tensor tBrB    = make_fragment_like<Quant>(tBsB);
    Tensor tBrB_dq = make_fragment_like(tBsB);
    Tensor tBgS    = thr_copy_b.partition_S(gS);
    Tensor tBrS    = make_fragment_like(tBgS(_,_,_,0));
    Tensor tBgZ    = thr_copy_b.partition_S(gZ);
    Tensor tBrZ    = make_fragment_like(tBgZ(_,_,_,0));
    Tensor tCrC    = thr_mma.make_fragment_C(tCgC);

    qmm_gemm<HasKResidue>(
        k_residue,
        copy_a, tApA, tAcA, tAgA, tArA, tAsA,
        copy_b, tBpB, tBcB, tBgB, tBrB, tBrB_dq, tBsB,
        tBgS, tBrS, tBgZ, tBrZ,
        mma, tCsA, tCsB, tCrC);

    // Store only the rows belonging to this group.
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      auto row = get<0>(tCcC(i));
      if ((row < m_max_coord) && (get<1>(tCcC(i)) < n_max_coord)) {
        if (row >= offset && row < offset_next) {
          tCgC(i) = Element(tCrC(i));
        }
      }
    }

    offset = offset_next;
  }
}

template <int TileM = 16, bool KMajor = true, bool HasKResidue = false, bool SM80 = true,
          typename Element, typename Quant, typename Scale>
void qmm_sorted_naive(
    const Element* A,
    const Quant*   B,
    const Scale*   S,
    const Element* Z,
    const uint32_t* rhs_indices,
    Element* C,
    int m, int n, int k, int e,
    auto group_size,
    auto&& launch_kernel) {
  // Define B stride and scales layout; e plays the role of the batch dim for B.
  auto dB = make_matrix_stride<KMajor>(n, k); // (dN,dK,dE)
  auto S_layout = make_scales_layout<KMajor>(n, k, e, group_size);

  // Define CTA tile sizes (static).
  auto bM = Int<TileM>{};
  auto bN = Int<(!SM80 && group_size > 64) ? 64 : 128>{};
  auto bK = Int<max(64, group_size)>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M,BLK_N,BLK_K)

  // Define MMA.
  TiledMMA mma = make_tiled_mma<TileM, SM80, Element>();
  auto num_threads = size(mma);

  // Define the A/B smem layouts (static).
  auto sA_layout = make_smem_layout(bM, bK);
  auto sB_layout = make_smem_layout<KMajor>(bN, bK);

  // Atoms.
  TiledCopy copy_a = make_tiled_copy<Element, true, HasKResidue>(num_threads, bM, bK);
  TiledCopy copy_b = make_tiled_copy<Quant, KMajor>(num_threads, bN, bK);

  auto* kernel = &qmm_sorted_naive_kernel<
      HasKResidue, decltype(cta_tiler),
      Element, Quant, Scale,
      decltype(dB), decltype(sA_layout), decltype(copy_a),
      decltype(sB_layout), decltype(copy_b),
      decltype(S_layout), decltype(mma)>;

  // Set L1 to be SMEM only.
  size_t smem_bytes = sizeof(SharedStorage<Element, decltype(sA_layout), decltype(sB_layout)>);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 num_blocks(size(ceil_div(m, bM)), size(ceil_div(n, bN)), 1);
  dim3 block_dims(num_threads);
  void* args[] = {
      &m, &n, &k, &e, &cta_tiler,
      &A, &sA_layout, &copy_a,
      &B, &dB, &sB_layout, &copy_b,
      &C,
      &S, &Z, &S_layout,
      &rhs_indices,
      &mma};
  launch_kernel(reinterpret_cast<void*>(kernel), num_blocks, block_dims, smem_bytes, args);
}

} // namespace cutlass_gemm

// clang-format on

namespace mlx::core {

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
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 32) {
    f.template operator()<32>();
  } else if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

template <typename T, typename F>
inline void dispatch_quant_types(
    int bits,
    int group_size,
    QuantizationMode mode,
    const char* tag,
    F&& f) {
  if (mode == QuantizationMode::Mxfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Mxfp8) {
    f.template operator()<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>();
  } else if (mode == QuantizationMode::Nvfp4) {
    f.template operator()<cutlass::float_e2m1_t, cutlass::float_e4m3_t, 16>();
  } else {
    dispatch_groups(group_size, tag, [&]<int group_size>() {
      if (bits == 2) {
        f.template operator()<cutlass::uint2b_t, T, group_size>();
      } else if (bits == 3) {
        f.template operator()<cutlass::uint3b_t, T, group_size>();
      } else if (bits == 4) {
        f.template operator()<cutlass::uint4b_t, T, group_size>();
      } else if (bits == 5) {
        f.template operator()<cutlass::uint5b_t, T, group_size>();
      } else if (bits == 6) {
        f.template operator()<cutlass::uint6b_t, T, group_size>();
      } else if (bits == 8) {
        f.template operator()<uint8_t, T, group_size>();
      } else {
        throw std::invalid_argument(
            fmt::format("{} {}-bit quantization is not supported.", tag, bits));
      }
    });
  }
}

template <int TileM, bool KMajor, bool HasKResidue, bool SM80>
void qmm_naive_impl(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[quantized_matmul]";
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  bool broadcast_b = (w.ndim() <= 2) || (w.size() != w.data_size());

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types<Element>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Quant, typename Scale, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          if (lhs_indices) {
            encoder.set_input_array(*lhs_indices);
          }
          if (rhs_indices) {
            encoder.set_input_array(*rhs_indices);
          }
          encoder.set_output_array(out);
          cutlass_gemm::qmm_naive<TileM, KMajor, HasKResidue, SM80>(
              gpu_ptr<Element>(x),
              gpu_ptr<Quant>(w),
              gpu_ptr<Scale>(scales),
              biases ? gpu_ptr<Element>(*biases) : nullptr,
              lhs_indices ? gpu_ptr<uint32_t>(*lhs_indices) : nullptr,
              rhs_indices ? gpu_ptr<uint32_t>(*rhs_indices) : nullptr,
              gpu_ptr<Element>(out),
              m,
              n,
              k,
              l,
              broadcast_b,
              cute::Int<group_size>{},
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  uint32_t smem_bytes,
                  void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, smem_bytes, args);
              });
        });
  });
}

template <int TileM, bool KMajor, bool HasKResidue, bool SM80>
void qmm_sorted_naive_impl(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  const char* tag = "[gather_qmm]";
  int k = x.shape(-1);
  int m = x.size() / k;
  int n = out.shape(-1);
  int e = w.size() / w.shape(-1) / w.shape(-2);

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types<Element>(
        bits,
        group_size,
        mode,
        tag,
        [&]<typename Quant, typename Scale, int group_size>() {
          encoder.set_input_array(x);
          encoder.set_input_array(w);
          encoder.set_input_array(scales);
          if (biases) {
            encoder.set_input_array(*biases);
          }
          encoder.set_input_array(rhs_indices);
          encoder.set_output_array(out);
          cutlass_gemm::qmm_sorted_naive<TileM, KMajor, HasKResidue, SM80>(
              gpu_ptr<Element>(x),
              gpu_ptr<Quant>(w),
              gpu_ptr<Scale>(scales),
              biases ? gpu_ptr<Element>(*biases) : nullptr,
              gpu_ptr<uint32_t>(rhs_indices),
              gpu_ptr<Element>(out),
              m,
              n,
              k,
              e,
              cute::Int<group_size>{},
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  uint32_t smem_bytes,
                  void** args) {
                encoder.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, smem_bytes, args);
              });
        });
  });
}

// clang-format off
template void qmm_naive_impl<@TileM@, @KMajor@, @HasKResidue@, @SM80@>(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

template void qmm_sorted_naive_impl<@TileM@, @KMajor@, @HasKResidue@, @SM80@>(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);
// clang-format on

} // namespace mlx::core

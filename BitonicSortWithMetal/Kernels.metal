//
//  Kernels.metal
//  BitonicSortWithMetal
//
//  Created by Takio Yamaoka on 2020/12/31.
//

#include <metal_stdlib>

using namespace metal;

//kernel void sortBitonicOnPhaseOverWarpKernel(const uint gid [[thread_position_in_grid]],
//                                             constant SortParameters &params [[buffer(0)]],
//                                             device uint32_t *data [[buffer(1)]],
//                                             threadgroup uint32_t *temp [[threadgroup(0)]])

#define SET_ALL(x) (~(!!(x))+1)

kernel void bitonicsortKernel(constant simd_uint2 &params [[buffer(0)]],
                              device uint32_t *data [[buffer(1)]],
                              const uint tid [[thread_position_in_grid]])
{
    const auto unit_size = params.x;   // size of bitonic
    const uint32_t block_size = params.y;  // size of comparison sets
    const uint32_t block_mask = block_size - 1;
    const auto no = tid & block_mask;   // comparator No. in block
    const auto index = ((tid & ~block_mask) << 1) | no;
    const auto l = data[index];
    const auto r = data[index | block_size];
#if 0
    const bool lt = l < r;
    const bool reverse = (index & unit_size) != 0;    // to toggle direction
    const simd_bool2 dir = simd_bool2(lt, !lt) ^ reverse;    // (lt, gte) or (gte, lt)
    const simd_uint2 v = select(simd_uint2(r), simd_uint2(l), dir);
    data[index] = v.x;
    data[index | block_size] = v.y;
#else
    const auto lt = l < r;
    const auto reverse = index & unit_size;
    if ( (!reverse && !lt) || (reverse && lt) ) {
        data[index] = r;
        data[index | block_size] = l;
    }
#endif
}

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
#if 0
    #define LOG2THREADGROUP(x)      ctz(x)
#else
    #define LOG2THREADGROUP(x)      10
#endif
#define SORT(F,L,R)                 \
    {                               \
        const auto v = sort(F,L,R); \
        (L) = v.x;                  \
        (R) = v.y;                  \
    }                               \

inline static constexpr int genLeftIndex(const uint tid, const uint32_t block_size)
{
    const uint32_t block_mask = block_size - 1;
    const auto no = tid & block_mask;   // comparator No. in block
    return ((tid & ~block_mask) << 1) | no;
}

inline static simd_uint2 sort(const bool reverse, uint32_t left, uint32_t right)
{
    const bool lt = left < right;
    const bool swap = !lt ^ reverse;
    const simd_bool2 dir = simd_bool2(swap, !swap);    // (lt, gte) or (gte, lt)
    const simd_uint2 v = select(simd_uint2(left), simd_uint2(right), dir);
    return v;
}

inline static void loadShared(const uint tgsize,
                              const uint sid,
                              const uint tid,
                              device uint32_t *data,
                              threadgroup uint32_t *shared)
{
    const auto index = genLeftIndex(tid, tgsize);
    shared[sid] = data[index];
    shared[sid | tgsize] = data[index | tgsize];
}

inline static void storeShared(const uint tgsize,
                               const uint sid,
                               const uint tid,
                               device uint32_t *data,
                               threadgroup uint32_t *shared)
{
    const auto index = genLeftIndex(tid, tgsize);
    data[index] = shared[sid];
    data[index | tgsize] = shared[sid | tgsize];
}

kernel void bitonicsortKernel(constant simd_uint2 &params [[buffer(0)]],    // x: monotonic width, y: comparative width
                              device uint32_t *data [[buffer(1)]],          // should be multiple of params.x
                              const uint tid [[thread_position_in_grid]])   // total threads should be half of data length
{
    const bool reverse = (tid & (params.x>>1)) != 0;    // to toggle direction
    const uint32_t block_size = params.y;  // size of comparison sets
    const auto left = genLeftIndex(tid, block_size);
    SORT(reverse, data[left], data[left | block_size]);
}

kernel void bitonicsortFirstRunKernel(constant simd_uint2 *params [[buffer(0)]],
                                      device uint32_t *data [[buffer(1)]],
                                      threadgroup uint32_t *shared [[threadgroup(0)]], // element num must be 2x (threads per threadgroup)
                                      const uint tgsize [[threads_per_threadgroup]],
                                      const uint simd_size [[threads_per_simdgroup]],
                                      const uint sid [[thread_index_in_threadgroup]],
                                      const uint tid [[thread_position_in_grid]])
{
#if 0
    const auto num = (LOG2THREADGROUP(tgsize) + 2) * (LOG2THREADGROUP(tgsize) + 1) >> 1;
    loadShared(tgsize, sid, tid, data, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < num; ++i) {
        const auto unit_size = params[i].x;
        const auto block_size = params[i].y;
        const bool reverse = (tid & (unit_size>>1)) != 0;
        const auto left = genLeftIndex(sid, block_size);
        SORT(reverse, shared[left], shared[left | block_size]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    storeShared(tgsize, sid, tid, data, shared);
#else
    loadShared(tgsize, sid, tid, data, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint unit_size = 1; unit_size <= tgsize; unit_size <<= 1) {
        const bool reverse = (tid & (unit_size)) != 0;    // to toggle direction
        for (uint block_size = unit_size; 0 < block_size; block_size >>= 1) {
            const auto left = genLeftIndex(sid, block_size);
            SORT(reverse, shared[left], shared[left | block_size]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    storeShared(tgsize, sid, tid, data, shared);
#endif
}

#if 0
kernel void bitonicsortInThreadGroupKernel(constant simd_uint2 &params [[buffer(0)]],
                                           device uint32_t *data [[buffer(1)]],
                                           threadgroup uint32_t *shared [[threadgroup(0)]], // element num must be 2x (threads per threadgroup)
                                           const uint tgsize [[threads_per_threadgroup]],
                                           const uint simd_size [[threads_per_simdgroup]],
                                           const uint sid [[thread_index_in_threadgroup]],
                                           const uint tid [[thread_position_in_grid]])
{
    loadShared(tgsize, sid, tid, data, shared);
    auto unit_size = params.x;
    auto block_size = params.y;
    if (simd_size < block_size || tgsize != block_size) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    do {
        const bool reverse = (tid & (unit_size>>1)) != 0;    // to toggle direction
        uint32_t blocks = block_size;
        do {
            const auto left = genLeftIndex(sid, blocks);
            SORT(reverse, shared[left], shared[left | blocks]);
            blocks >>= 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } while (0 < blocks);
        block_size = unit_size;
        unit_size <<= 1;
    } while (block_size <= tgsize);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    storeShared(tgsize, sid, tid, data, shared);
}

#else
kernel void bitonicsortInThreadGroupKernel(constant simd_uint2 &params [[buffer(0)]],
                                           device uint32_t *data [[buffer(1)]],
                                           threadgroup uint32_t *shared [[threadgroup(0)]], // element num must be 2x (threads per threadgroup)
                                           const uint tgsize [[threads_per_threadgroup]],
                                           const uint simd_size [[threads_per_simdgroup]],
                                           const uint sid [[thread_index_in_threadgroup]],
                                           const uint tid [[thread_position_in_grid]])
{
    loadShared(tgsize, sid, tid, data, shared);
    const auto unit_size = params.x;
    const auto block_size = params.y;
    const auto num = LOG2THREADGROUP(tgsize) + 1;
    const bool reverse = (tid & (unit_size>>1)) != 0;    // to toggle direction
    for (uint i = 0; i < num; ++i) {
        const auto width = block_size >> i;
        const auto left = genLeftIndex(sid, width);
        SORT(reverse, shared[left], shared[left | width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    storeShared(tgsize, sid, tid, data, shared);
}
#endif

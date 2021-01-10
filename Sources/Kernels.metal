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

inline static int genLeftIndex(const uint tid, const uint32_t block_size)
{
    const uint32_t block_mask = block_size - 1;
    const auto no = tid & block_mask;   // comparator No. in block
    return ((tid & ~block_mask) << 1) | no;
}

inline static void sort(const bool reverse, device uint32_t &left, device uint32_t &right)
{
    const auto l = left;
    const auto r = right;
    const bool lt = l < r;
#if 1
    const simd_bool2 dir = simd_bool2(lt, !lt) ^ reverse;    // (lt, gte) or (gte, lt)
    const simd_uint2 v = select(simd_uint2(r), simd_uint2(l), dir);
    left = v.x;
    right = v.y;
#else
    if ( (!reverse && !lt) || (reverse && lt) ) {
        left = r;
        right = l;
    }
#endif
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
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

kernel void bitonicsortKernel(constant simd_uint2 &params [[buffer(0)]],
                              device uint32_t *data [[buffer(1)]],
                              const uint tid [[thread_position_in_grid]])
{
    const bool reverse = (tid & (params.x>>1)) != 0;    // to toggle direction
    const uint32_t block_size = params.y;  // size of comparison sets
    const auto left = genLeftIndex(tid, block_size);
    sort(reverse, data[left], data[left | block_size]);
}

kernel void bitonicsortFirstRun(constant simd_uint2 &params [[buffer(0)]],
                                device uint32_t *data [[buffer(1)]],
                                threadgroup uint32_t *shared [[threadgroup(0)]], // element num must be 2x (threads per threadgroup)
                                const uint tgsize [[threads_per_threadgroup]],
                                const uint sid [[thread_index_in_threadgroup]],
                                const uint tid [[thread_position_in_grid]])
{
    loadShared(tgsize, sid, tid, data, shared);
    auto unit_size = params.x;
    auto block_size = params.y;
    do {
        const bool f_reverse = (tid & (unit_size>>1)) != 0;    // to toggle direction
        uint32_t blocks = block_size;
        do {
            const auto left = genLeftIndex(sid, blocks);
            const auto l = shared[left];
            const auto r = shared[left | blocks];
            const bool lt = l < r;
            const simd_bool2 dir = simd_bool2(lt, !lt) ^ f_reverse;    // (lt, gte) or (gte, lt)
            const simd_uint2 v = select(simd_uint2(r), simd_uint2(l), dir);
            shared[left] = v.x;
            shared[left | blocks] = v.y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            blocks >>= 1;
        } while (0 < blocks);
        block_size = unit_size;
        unit_size <<= 1;
    } while (block_size <= tgsize);
    storeShared(tgsize, sid, tid, data, shared);
}

kernel void bitonicsortInThreadGroupKernel(constant simd_uint2 &params [[buffer(0)]],
                                           device uint32_t *data [[buffer(1)]],
                                           threadgroup uint32_t *shared [[threadgroup(0)]], // element num must be 2x (threads per threadgroup)
                                           const uint tgsize [[threads_per_threadgroup]],
                                           const uint sid [[thread_index_in_threadgroup]],
                                           const uint tid [[thread_position_in_grid]])
{
    loadShared(tgsize, sid, tid, data, shared);
    auto unit_size = params.x;
    auto block_size = params.y;
    do {
        const bool f_reverse = (tid & (unit_size>>1)) != 0;    // to toggle direction
        uint32_t blocks = block_size;
        do {
            const auto left = genLeftIndex(sid, blocks);
            const auto l = shared[left];
            const auto r = shared[left | blocks];
            const bool lt = l < r;
            const simd_bool2 dir = simd_bool2(lt, !lt) ^ f_reverse;    // (lt, gte) or (gte, lt)
            const simd_uint2 v = select(simd_uint2(r), simd_uint2(l), dir);
            shared[left] = v.x;
            shared[left | blocks] = v.y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            blocks >>= 1;
        } while (0 < blocks);
        block_size = unit_size;
        unit_size <<= 1;
    } while (block_size <= tgsize);
    storeShared(tgsize, sid, tid, data, shared);
}

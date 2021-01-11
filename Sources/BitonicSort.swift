//
//  BitonicSort.swift
//  BitonicSortWithMetal
//
//  Created by Takio Yamaoka on 2021/01/03.
//

import Foundation
import Metal
import simd

class BitonicSort {
    private let device: MTLDevice
    private let cmdQueue: MTLCommandQueue
    private let bitonicsortState: MTLComputePipelineState
    private let bitonicsortFirstRunState: MTLComputePipelineState
    private let bitonicsortInThreadGroupState: MTLComputePipelineState

    private var cmdBuffer: MTLCommandBuffer?
    private var cmdEncoder: MTLComputeCommandEncoder?
    
    private var mtlParameter: MTLBuffer
    private var mtlBuffer: MTLBuffer?
    private var count: Int?
    private var length: Int?

    var use_threadgroup = true

    init?(device: MTLDevice, library: MTLLibrary) {
        guard let f1 = library.makeFunction(name: "bitonicsortKernel"),
              let f2 = library.makeFunction(name: "bitonicsortFirstRunKernel"),
              let f3 = library.makeFunction(name: "bitonicsortInThreadGroupKernel"),
              let k1 = try? device.makeComputePipelineState(function: f1),
              let k2 = try? device.makeComputePipelineState(function: f2),
              let k3 = try? device.makeComputePipelineState(function: f3),
              let q = device.makeCommandQueue()
        else {
            return nil
        }
        self.device = device
        self.cmdQueue = q
        self.bitonicsortState = k1
        self.bitonicsortFirstRunState = k2
        self.bitonicsortInThreadGroupState = k3
        // make parameter table for first run
        let target = self.bitonicsortFirstRunState.maxTotalThreadsPerThreadgroup
        var v = simd_uint2(repeating: 1)
        var params: [simd_uint2] = []
        while v.x <= target {
            v.y = v.x
            v.x <<= 1
            while 0 < v.y {
                params.append(v)
                v.y >>= 1
            }
        }
        guard let buffer = device.makeBuffer(
                length: MemoryLayout<simd_uint2>.stride * params.count,
                options: [.storageModeShared])
        else {
            return nil
        }
        let raw = buffer.contents()
        raw.initializeMemory(as: simd_uint2.self, from: &params, count: params.count)
        self.mtlParameter = buffer
    }
    
    convenience init?() {
        guard let d = MTLCreateSystemDefaultDevice(),
              let l = d.makeDefaultLibrary()
        else {
            return nil
        }
        self.init(device: d, library: l)
    }
    
    func setData(array: inout [UInt32]) {
        let length = 1 << UInt(ceil(log2f(Float(array.count))))
        guard let buffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * length,
                options: [.storageModeShared])
        else {
            return
        }
        let raw = buffer.contents()
        raw.initializeMemory(as: UInt32.self, from: array, count: array.count)
        if array.count < length {
            let left = raw.advanced(by: MemoryLayout<UInt32>.stride * array.count)
            left.initializeMemory(as: UInt32.self, repeating: UInt32.max, count: length - array.count)
        }
        self.count = array.count
        self.length = length
        self.mtlBuffer = buffer
    }
    
    func getPointer() -> (UnsafePointer<UInt32>, Int)? {
        guard let raw = mtlBuffer?.contents() else {
            return nil
        }
        let p = UnsafePointer(raw.assumingMemoryBound(to: UInt32.self))
        return (p, count!)
    }
    
    func prepare() {
        cmdBuffer = cmdQueue.makeCommandBuffer()
        cmdEncoder = cmdBuffer?.makeComputeCommandEncoder()
    }
    
    func commit() {
        cmdEncoder?.endEncoding()
        cmdBuffer?.commit()
        cmdEncoder = nil
    }
    
    func wait() {
        cmdBuffer?.waitUntilCompleted()
    }
    
    public func sort() {
        guard let enc = cmdEncoder,
              let buf = mtlBuffer
        else {
            return
        }
        let grid_size = MTLSizeMake(length! >> 1, 1, 1)
        let unit_size = min(grid_size.width, bitonicsortState.maxTotalThreadsPerThreadgroup)
        let group_size = MTLSizeMake(unit_size, 1, 1)
        var params = simd_uint2(repeating: 1)
        // first run
        if use_threadgroup {
            params.x = UInt32(unit_size << 1)
            enc.setComputePipelineState(bitonicsortFirstRunState)
            enc.setBuffer(mtlParameter, offset: 0, index: 0)
            enc.setBuffer(buf, offset: 0, index: 1)
            enc.setThreadgroupMemoryLength((MemoryLayout<UInt32>.stride * unit_size) << 1, index: 0)
            enc.dispatchThreads(grid_size, threadsPerThreadgroup: group_size)
        }
        while params.x < length! {
            params.y = params.x
            params.x <<= 1
            repeat {
                if !use_threadgroup || unit_size < params.y {
                    enc.setComputePipelineState(bitonicsortState)
                    enc.setBytes(&params, length: MemoryLayout<simd_uint2>.stride, index: 0)
                    enc.setBuffer(buf, offset: 0, index: 1)
                    params.y >>= 1
                }
                else {
                    enc.setComputePipelineState(bitonicsortInThreadGroupState)
                    enc.setBytes(&params, length: MemoryLayout<simd_uint2>.stride, index: 0)
                    enc.setBuffer(buf, offset: 0, index: 1)
                    enc.setThreadgroupMemoryLength((MemoryLayout<UInt32>.stride * unit_size) << 1, index: 0)
//                    params.x = max(UInt32(unit_size << 1), params.x)
                    params.y = 0
                }
                enc.dispatchThreads(grid_size, threadsPerThreadgroup: group_size)
            } while 0 < params.y
        }
    }
}

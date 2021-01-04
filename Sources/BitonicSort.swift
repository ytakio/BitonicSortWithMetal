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
    
    private var cmdBuffer: MTLCommandBuffer?
    private var cmdEncoder: MTLComputeCommandEncoder?
    
    private var mtlBuffer: MTLBuffer?
    private var length: Int?

    init?(device: MTLDevice, library: MTLLibrary) {
        guard let f = library.makeFunction(name: "bitonicsortKernel"),
              let k = try? device.makeComputePipelineState(function: f),
              let q = device.makeCommandQueue()
        else {
            return nil
        }
        self.device = device
        self.cmdQueue = q
        self.bitonicsortState = k
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
        self.mtlBuffer = buffer
        self.length = length
    }
    
    func getPointer() -> (UnsafePointer<UInt32>, Int)? {
        guard let raw = mtlBuffer?.contents() else {
            return nil
        }
        let p = UnsafePointer(raw.assumingMemoryBound(to: UInt32.self))
        return (p, length!)
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
        var params = simd_uint2(repeating: 1)
        while params.x < length! {
            params.x <<= 1
            params.y = params.x >> 1
            repeat {
                enc.setComputePipelineState(bitonicsortState)
                enc.setBytes(&params, length: MemoryLayout<simd_uint2>.stride, index: 0)
                enc.setBuffer(buf, offset: 0, index: 1)
                enc.dispatchThreads(grid_size,
                                    threadsPerThreadgroup: MTLSizeMake(unit_size, 1, 1))
                params.y >>= 1
            } while 0 < params.y
        }
    }
}

//
//  Utils.swift
//  BitonicSortWithMetal
//
//  Created by Takio Yamaoka on 2021/01/04.
//

import Foundation

let NUM = 1 << 22

func setRandom(_ data: inout [UInt32]) {
    for i in 0 ..< data.count {
        data[i] = UInt32.random(in: 0 ..< UInt32(data.count))
    }
}

func checkData(d1: UnsafePointer<UInt32>, d2: UnsafePointer<UInt32>, count: Int)
-> Bool {
    var bool = true
    for i in 0 ..< count {
        if d1[i] != d2[i] {
            bool = false
            print(i, ":d1=", d1[i], "d2=", d2[i])
        }
    }
    return bool
}

func testSort(_ algorithm: BitonicSort) -> String {
    var log = String()
    log += "start\n"
    var data = Array<UInt32>(repeating: 0, count: NUM)
    // without threadgroup_barrier
    setRandom(&data)
    algorithm.setData(array: &data)
    let metal_start = Date()
    algorithm.prepare()
    algorithm.sort()
    algorithm.commit()
    algorithm.wait()
    let metal_end = Date()
    let cpu_start = Date()
    data.sort()
    let cpu_end = Date()
    guard let (mtl, count) = algorithm.getPointer() else {
        log += "can't get device memory pointer!!\n"
        return log
    }
    if checkData(d1: data, d2: mtl, count: count) {
        log += "sorting \(data.count) elements passed.\n"
        log += "metal time: " + String(metal_start.distance(to: metal_end)) + "\n"
        log += "cpu time: " + String(cpu_start.distance(to: cpu_end)) + "\n"
    }
    else {
        log += "fail!\n"
    }
    log += "finished\n"
    return log
}

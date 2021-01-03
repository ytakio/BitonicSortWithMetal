//
//  ViewController.swift
//  BitonicSortWithMetal
//
//  Created by Takio Yamaoka on 2021/01/03.
//

import UIKit
import Metal

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

class ViewController: UIViewController, UITextViewDelegate {
    let NUM = 1_000_000

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        guard let tv = view.subviews[0] as? UITextView else {
            exit(-1)
        }
        tv.text += "start\n"
        var data = Array<UInt32>(repeating: 0, count: NUM)
        setRandom(&data)
        guard let bitonic = BitonicSort() else {
            tv.text += "instantiation failed!!\n"
            return
        }
        bitonic.setData(array: &data)
        let metal_start = Date()
        bitonic.prepare()
        bitonic.sort()
        bitonic.commit()
        bitonic.wait()
        let cpu_start = Date()
        data.sort()
        let end_time = Date()
        guard let (mtl, _) = bitonic.getPointer() else {
            tv.text += "can't get device memory pointer!!\n"
            return
        }
        if checkData(d1: data, d2: mtl, count: data.count) {
            tv.text += "sorting \(data.count) elements passed.\n"
            tv.text += "metal time: " + String(metal_start.distance(to: cpu_start)) + "\n"
            tv.text += "cpu time: " + String(cpu_start.distance(to: end_time)) + "\n"
        }
        else {
            tv.text += "fail!\n"
        }
    }


}


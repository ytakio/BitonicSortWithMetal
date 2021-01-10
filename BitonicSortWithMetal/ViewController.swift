//
//  ViewController.swift
//  BitonicSortWithMetal
//
//  Created by Takio Yamaoka on 2021/01/03.
//

import UIKit
import Metal

class ViewController: UIViewController, UITextViewDelegate {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        guard let tv = view.subviews[0] as? UITextView else {
            exit(-1)
        }
        // test
        var log = String()
        guard let bitonic = BitonicSort() else {
            log += "instantiation failed!!\n"
            tv.text = log
            return
        }
        log += "======= without threadgroup_barrier =======\n"
        bitonic.use_threadgroup = false
        log += testSort(bitonic)
        log += "======= start using threadgroup_barrier =======\n"
        bitonic.use_threadgroup = true
        log += testSort(bitonic)
        tv.text = log
        print(log)
    }

}


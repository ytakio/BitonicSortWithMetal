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
        tv.text = testSort()
    }

}


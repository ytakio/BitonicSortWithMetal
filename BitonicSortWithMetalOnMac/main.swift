//
//  main.swift
//  BitonicSortWithMetalOnMac
//
//  Created by Takio Yamaoka on 2021/01/04.
//

import Foundation

var log = String()
guard let bitonic = BitonicSort() else {
    log += "instantiation failed!!\n"
    print(log)
    exit(-1)
}
log = "======= without threadgroup_barrier =======\n"
bitonic.use_threadgroup = false
log += testSort(bitonic)
print(log)
log = "======= start using threadgroup_barrier =======\n"
bitonic.use_threadgroup = true
log += testSort(bitonic)
print(log)

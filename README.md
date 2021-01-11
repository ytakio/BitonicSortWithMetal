# BitonicSortWithMetal

- iPhone12 Pro
```
======= without threadgroup_barrier =======
start
sorting 4194304 elements passed.
metal time: 0.21709799766540527
cpu time: 0.43490707874298096
finished
======= start using threadgroup_barrier =======
start
sorting 4194304 elements passed.
metal time: 0.09405803680419922
cpu time: 0.35382699966430664
finished
```

- Macbooc Pro 13 (2020)
```
======= without threadgroup_barrier =======
start
sorting 4194304 elements passed.
metal time: 0.19690394401550293
cpu time: 0.3822319507598877
finished

======= start using threadgroup_barrier =======
start
sorting 4194304 elements passed.
metal time: 0.11200106143951416
cpu time: 0.37771105766296387
finished
```

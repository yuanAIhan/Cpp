#### atomicAdd

​	这个函数是先进行加法计算之后，然后再进行赋值的！

下面是其的源码实现：

```cpp
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, doble val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = * address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + _longlong_as_double(assumed)));
    } while(assumed != old);
    return __longlong_as_double(old);
}
```

​    atomicAdd实际是调用atomicCAS函数，atomicSub、atomicMax等函数都是如此，了解一个就都明白了。从代码中看到，atomicCAS是对地址操作，将结果存在地址上，返回的是还是old值。

实际上对应的atomicCAS的函数本质是：

```cpp
old == compare ? val : old;
```


#### **CUDA 语言编程一些简单总结：**

第一个cuda程序hello world 

​		修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行,编写一个内核函数，命名为helloFromGPU:

```c
__global__ void helloFromGPU(void)
{
    printf("Hello World from CPU!\n");
}
```

​	用下面的代码启动内核函数

```c
helloFromGPU <<<1, 10 >>>();
```

​	**三重尖括号意味着从主线程到设备端代码的调用。一个内核函数通过一组线程来执行，所有线程执行相同的代码。三重尖括号里面的参数是执行配置，用来说明使用多少线程来执行内核函数。**在这个例子中，有10个GPU线程被调用。

```c
#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from CPU!\n");
}
int main(void)
{
    printf("Hello World from GPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
}
```

​	函数cudaDeviceRest（）用来显式地释放和清空当前进程中与当前设备有关的所有资源。

编译对应的代码的时候，可以直接使用nvcc的方式来完成，有的需要按照具体的设备来完成！最后的得到的结果的为：

 nvcc hello.cu -o hello
 ./hello 
Hello World from CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!

一般而言，一个典型的CUDA程序编程结构应该包含由五个主要步骤：

- 1.分配GPU内存。
- 2.从CPU内存中拷贝数据到GPU内存。
- 3.调用CUDA内核函数来完成程序指定的运算
- 4.将数据从GPU拷回CPU内存。
- 5.释放GPU内存空间。

   CUDA是一种通用的并行计算平台和编程模型，是在C语言基础上扩展的。借助于CUDA，你可以像编写C语言程序一样实现并行算法。你可以在NVIDIA的GPU平台上用CUDA为多种系统编写应用程序，范围从嵌入式设备、平板电脑、笔记本电脑、台式机、工作站到HPC集群（高性能计算集群）。





#### 一种通过层次结构在GPU中组织线程的方法：

#### 	 因为开发一个项目的原因，需要使用相关的内存管理和设备之间数据传输的代码，所以在这里记录一下！

​	主机内存和设备内存是不同的，对应的开发的时候使用的是完全不相同的、需要注意。从CUDA 6.0开始，NVIDIA提出了名为“统一寻址”（Unified Memory）的编程模型的改进，它连接了主机内存和设备内存空间，可使用单个指针访问CPU和GPU内存，无须彼此之间手动拷贝数据。重要的是应学会如何为主机和设备分配内存空间以及如何在CPU和GPU之间拷贝共享数据。

​	内核是CUDA编程模型的一个重要的组成部分，其代码是在GPU上运行的。主机的代码使用C语言进行编写即可，但是设备代码必须使用CUDA进行编写，当然是可以将对应的全部的代码放在一个源文件下面的。

一个典型的CUDA程序实现流程遵循以下模式。

- 1.把数据从CPU内存拷贝到GPU内存。
- 2.调用核函数对存储在GPU内存中的数据进行操作
-  3.将数据从GPU内存传送回到CPU内存。

#### 2.2 内存管理：

​	CUDA编程模型假设系统是由一个主机和一个设备组成的，而且各自拥有独立的内存。核函数是在设备上运行的。为使你拥有充分的控制权并使系统达到最佳性能，**CUDA运行时负责分配与释放设备内存，并且在主机内存和设备内存之间传输数据**。

​	1. 用于执行GPU内存分配的是cudaMalloc函数，其函数原型为：

```c
cudaError_t cudaMalloc (void** devPtr, size_t size)
////-----------------------------------------一些函数
    
/*/// 前者是C语言相关的一些内存操作的函数，后者是cuda相关的内存操作的函数！
     malloc    ---- cudaMalloc
     free      ---- cudaFree
     memset    ---- cudaMemset
     memcpy    ---- cudaMemcpy
*/    
```

  其中函数： cudaMalloc 该函数负责向设备分配一定字节的线性内存，并以devPtr的形式返回指向所分配内存的指针.cudaMalloc与标准C语言中的malloc函数几乎一样，只是此函数在GPU的内存里分配内存。通过充分保持与标准C语言运行库中的接口一致性，可以实现CUDA应用程序的轻松接入。

​	2. cudaMemcpy函数负责主机和设备之间的数据传输，其函数原型为：

```c
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    //其中cudaError_t 是一种返回类型，成功的时候返回为cudaSuccess 否则返回具体情况的错误信息！
```

​	此函数从src指向的源存储区复制一定数量的字节到dst指向的目标存储区。复制方向由kind指定，其中的kind有以下几种。

1. cudaMemcpyHostToHost
2. cudaMemcpyHostToDevice
3. cudaMemcpyDeviceToHost
4. cudaMemcpyDeviceToDevice

​     这个函数以同步方式执行，因为在cudaMemcpy函数返回以及传输操作完成之前主机应用程序是阻塞的。除了内核启动之外的CUDA调用都会返回一个错误的枚举类型cudaError_t。**如果GPU内存分配成功，函数返回 cudaSuccess 否则返回是：cudaErrorMemoryAllocation**。



#### 内存层次结构:

​	在GPU内存层次结构中，**最主要的两种内存是全局内存和共享内存。全局类似于CPU的系统内存，而共享内存类似于CPU的缓存**。然而GPU的共享内存可以由CUDA C的内核直接控制。



下面，我们将通过一个简单的两个数组相加的例子来学习如何在主机和设备之间进行数据传输，以及如何使用CUDA C编程。
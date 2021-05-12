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

![image-20210511205753257](C:\Users\guiyuan\AppData\Roaming\Typora\typora-user-images\image-20210511205753257.png)

下面，我们将通过一个简单的两个数组相加的例子来学习如何在主机和设备之间进行数据传输，以及如何使用CUDA C编程。

```c
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
    for(int idx = 0; idx < N; idx ++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrayOnGpu(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main()
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    int  flag = 1;
    if(flag)
    {
        float *h_a, *h_b, *h_c;
        h_a = (float*)malloc(nBytes);
        h_b = (float*)malloc(nBytes);
        h_c = (float*)malloc(nBytes);
        initialData(h_a, nElem);
        initialData(h_b, nElem);
        sumArrayOnHost(h_a, h_b, h_c, nElem);
        free(h_a);
        free(h_b);
        free(h_c);
    }
    else 
    {
        float *h_a, *h_b, *h_c;
        h_a = (float*)malloc(nBytes);
        h_b = (float*)malloc(nBytes);
        h_c = (float*)malloc(nBytes);
        initialData(h_a, nElem);
        initialData(h_b, nElem);

        float *d_a, *d_b, *d_c;
        cudaMalloc((float**)&d_a, nBytes);
        cudaMalloc((float**)&d_b, nBytes);
        cudaMalloc((float**)&d_c, nBytes);

        cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    return 0;
}
```

#### 线程管理：

​	当核函数在主机端启动时，它的执行会移动到设备上，此时设备中会产生大量的线程并且每个线程都执行由核函数指定的语句。**CUDA明确了线程层次抽象的概念以便于你组织线程**。这是一个两层的线程层次结=构，由线程块和线程块网格构成，如图2-5所示。

​	                              ![image-20210512095924690](C:\Users\guiyuan\AppData\Roaming\Typora\typora-user-images\image-20210512095924690.png)

**由一个内核启动所产生的全部线程统称为一个网格，同一个网格中所有的线程共享相同的全局内存空间。**

一个网格是由多个线程块构成的，一个线程块包含一组线程，同一线程块内的线程协作可以通过下面的方式实现：

​	同步和共享内存。不同块内的线程是不能协作的.线程依靠一下两个坐标变量来区分彼此：

- blockIdx  线程块在线程格内的索引
- threadIdx 块内的线程索引

​    这些变量是核函数中需要预初始化的内置变量。当执行一个核函数时，CUDA运行时为每个线程分配坐标变量blockIdx和threadIdx。基于这些坐标，你可以将部分数据分配给不同的线程。

​	该坐标变量是基于uint3定义的CUDA内置的向量类型，是一个包含3个无符号整数的结构，可以通过x、y、z三个字段来指定。

```c
blockIdx.x
blockIdx.y
blockIdx.z

threadIdx.x
threadIdx.y
threadIdx.z
```

​	网格和块的维度由下列两个内置变量指定：

```c
blockDim //线程块的维度，用每个线程块中线程数来表示
gridDim //线程格的维度，用每个线程中线程数来表示

blockDim.x
blockDim.y
blockDim.z
    
    
//通常，一个线程格会被组织成线程块的二维数组形式，一个线程块会被组织成线程的三维数组形式
    
//线程格和线程块均使用3个dim3类型的无符号整型字段，而未使用的字段将被初始化为1且忽略不计。
```

​	在CUDA程序中有两组不同的网格和块变量：手动定义的dim3数据类型和预定义的uint3数据类型。在主机端，作为内核调用的一部分，你可以使用dim3数据类型定义一个网格和块的维度。当执行核函数时，CUDA运行时会生成相应的内置预初始化的网格、块和线程变量，它们在核函数内均可被访问到且为unit3类型。手动定义的dim3类型的网格和块变量仅在主机端可见，而unit3类型的内置预初始化的网格和块变量仅在设备端可见。

```c
int nElem = 6;

dim3 block(3);
dim3 grid((nElem + block.x -1 ) / block.x);

```

```c
#include <stdio.h>

int nElem = 6;

dim3 block(3);
dim3 grid((nElem + block.x - 1) / block.x);

int main(void)
{
    //在主机端上的程序段用来检查网格和块维度！
    printf("grid.x %d  gird.y  %d  grid.z %d \n", grid.x, grid.y, grid.z);
    printf("block.x %d  block.y  %d  block.z %d \n", block.x, block.y, block.z);
    return 0;
}
```

汇总代码如下：

```c#
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void chechIndex(void)
{
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d,%d,%d) blockdim:(%d,%d,%d)"
            "gridDim:(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x,blockIdx.y,blockIdx.z, blockDim.x,blockDim.y,blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d  gird.y  %d  grid.z %d \n", grid.x, grid.y, grid.z);
    printf("block.x %d  block.y  %d  block.z %d \n", block.x, block.y, block.z);

    chechIndex<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}

/*
grid.x 2  gird.y  1  grid.z 1 
block.x 3  block.y  1  block.z 1 
threadIdx:(0, 0, 0) blockIdx:(1,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
threadIdx:(1, 0, 0) blockIdx:(1,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
threadIdx:(2, 0, 0) blockIdx:(1,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
threadIdx:(0, 0, 0) blockIdx:(0,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
threadIdx:(1, 0, 0) blockIdx:(0,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
threadIdx:(2, 0, 0) blockIdx:(0,0,0) blockdim:(3,1,1)gridDim:(2,1,1)
*/
```

- 确定块的大小
- 在已知数据大小和块大小的基础上计算网格维度

要确定块尺寸，一般需要考虑：

- 内核的性能特性
- GPU资源的限制

#### 启动一个核函数

```ｃ
// kernel_function <<<4, 8>>>(argument list);
```

![image-20210512112331297](C:\Users\guiyuan\AppData\Roaming\Typora\typora-user-images\image-20210512112331297.png)

核函数中前者指定的是对应的使用块的个数，后者是一个块中对应的元素个数值。对应的线程布局方式如上所示：可以借助于这个理解对应的参数的使用！

编写核函数：

![image-20210512113049290](C:\Users\guiyuan\AppData\Roaming\Typora\typora-user-images\image-20210512113049290.png)

　函数类型限定符指定一个函数在主机上执行还是在设备上执行，以及可被主机调用还是被设备调用。

CUDA核函数限制：

- 只可以访问设备内存
- 必须具有void返回类型
- 不支持可变数量的参数
- 不支持静态变量
- 显示异步行为

**验证核函数**：其次，可以将执行参数设置为<<<1，1>>>，因此强制用一个块和一个线程执行核函数，这模拟了串行执行程序。这对于调试和验证结果是否正确是非常有用的，而且，如果你遇到了运算次序的问题，这有助于你对比验证数值结果是否是按位精确的。

#### 处理错误：

​	
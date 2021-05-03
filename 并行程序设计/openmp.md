**并行计算**：

  OPENMP的简单原理：

参考博客：https://blog.csdn.net/dulingwen/article/details/103711344

​	https://www.cnblogs.com/liangliangh/p/3565136.html

- ​	编译指导语句
- ​    库函数
- ​    环境变量
- ​    数据类型
- ​    宏定义

  可以简单的分为共享内存（多个cpu共享一个内存）和分布式内存，主要的思想就是使用多线程，让可以在并行负载分配到多个物理计算核心上。便可以完成缩短执行时间，同时提高CPU利用率。fork/join 式并行！

- parallel : 多个线程的执行顺序是不能保证的！这个可以理解位对应的会生成一个并行的多个线程来执行对应的程序。omp_get_thread_num()显然这个函数就是获取当前执行的进程数值的函数！

```cpp
int a = 0;
#pragma omp parallel if(a) num_threads(6)
{
    std::cout << omp_get_thread_num();
}

```

- for: 对于for循环的一个操作实现！for循环的使用必须是对应的for循环的执行顺序不会影响最终的结果！否则其就会报错的出现！

```cpp
const int size = 1000;
int data[size];
#pragma omp parallel //执行到这里的时候会派生出来7个线程1
{
    #pragma omp for
    for(int i = 0; i < size; i++)
        data[i] = 100;
}
//简写为下面的代码：
#pragma omp parallel for 
for(int i=0; i < size; i++){
    data[i] = 1000;
}

```

- schedule(type, [,size]):这个用来实现对应的进程组的划分。也就是按照一定的规则将其中的不同的进程进行一个划分。

```cpp
#pragma omp parallel num_threads(3)
{
    #pragma omp for
    for(int i=0; i<9; ++i){
        #pragma omp critical
        std::cout << omp_get_thread_num() << i << " ";
    }
}

 #pragma omp parallel num_threads(3)
 {
     #pragma omp for schedule(static, 1)
     for(int i=0; i<9; ++i){
         #pragma omp critical 
         std::cout << omp_get_thread_num() << i << " ";
     }
 }
 //也就是说指定对应的执行空间的线程分配，将对应的for循环指定的对应的全部数值进行一个划分实现！
```

- sections:任务并行，指示后面的代码块包含将被多个线程并行执行的section块！

```cpp
#pragma omp parallel
{
    #pragma mop sections
    {
        #pragma omp section 
        std::cout << omp_get_thread_num();
        #pragma omp section
        std::cout << omp_get_thread_num();
    }
}

#pragma omp parallel sections
```

- single:指示当前代码只能被一个线程执行，具体哪个线程不确定！

```
#pragma omp parallel num_threads(4)
{
	#pragma omp single 
	std::cout << omp_get_thread_num();
	std::cout << " ";
}
```

- master: 指示代码只能被主线程执行，功能类似于single，但是single是无法得知当前的哪个线程执行对应的代码的！

- critical：定义一个临界区，保证同一时刻只有一个线程访问临界区：上面的代码执行的时候可能会导致每次输出的结果是不同的，但是后面的代码的输出的结果永远都是前后的结果一样的，是不会发生任何改变的！

```cpp
#pragma omp parallel num_threads(6)
{
    std::cout << omp_get_thread_num() << omp_get_thread_num();
}

#pragma omp parallel num_threads(6)
{
    #pragma omp critical
    std::cout << omp_get_thread_num() << omp_get_thread_num();
}
```

- barrier:定义一个同步，所有线程都执行到改行的时候，所有的线程才开始执行后续的代码。同时上面的代码中for、sections、single、directives中隐含有哦barrier的值，以及 nowait 的值都具有对应的实列存在。

```
#pragma omp parallel num_threads(6)
{
	#pragma omp critical
	std::cout << omp_get_thread_num() << " ";
	#pragma omp critical 
	std::cout << omp_get_thread_num() + 10 << " ";
}

#pragma omp parallel num_threads(6)
{
	#pragma omp critical 
	std::cout << omp_get_thread_num() << " ";
	#pragma omp barrier 
	#pragma omp critical
	std::cout << omp_get_thread_num() + 10 << " ";
}
```

- atomic:保证变量被原子的更新，即同意额时刻只有一个线程在更新该变量！

```cpp
int m = 0 ;
#pragma omp parallel num_threads(6)
{
	for(int i=0; i < 10000; i++) {
		m++;
	}
}
std::cout << m << std::endl;  //得到的值不是600000

int m = 0 ;
#pragma omp parallel num_threads(6)
{
	for(int i=0; i < 10000; i++) {
        #pragma omp atomic //  #pragma omp critical
		m++;
	}
}
std::cout << m << std::endl; //得到的值是对应的60000


```

- flush：所有线程对多有共享对象具有相同的内存视图，将对变量的更新直接协会内存中去的！对于一些场景中对应的将当前修改的值，每次修改之后都直接写回到内存之中、保证有其它程序的时候可以很方便的取出来！

```cpp
int data = 0, flag = 0;
#pragma omp parallel sections num_threads(2) shared(data, flag)
{
    #pragma omp section 
    {
        #pragma omp critical
        std::cout << "thread" << omp_get_thread_num() << std::endl;
    	#pragma omp flush(data)
        flag = 1;
        #pragma omp flush(flag) //第10行代码告诉编译器，确保data的新值已经写回内存
    }
    #pragma omp scetion
    {
        while(!flag){
            #pragma omp flush(flag)
        }
        #pragma omp critical
        std::cout << "thread" << omp_get_thread_num() << std::endl;
        #pragma omp flush(data)
        --data
        std::cout << data << std::endl;
    }
}
```

- ordered:确保当前的代码可以按照迭代的次数执行（像对应的串行程序一样的）

```cpp
#pragma omp parallel num_threads(8)
{
    #pragma omp for ordered
    for(int i=0; i<10; i++) {
        #pragma omp critical
         std::cout << i << " ";
        #pragma omp ordered //将顺序的执行1
        {
            #pragma omp critical
            	std::cout << "-" << i << " ";
        }
    }
}
```

- thread private: 将全局或静态变量声明为线程私有的。

```cpp
#include <omp.h>
#include <iostream>
int a;

#pragma omp threadprivate(a)
int main()
{
	std::cout << omp_get_thread_num() << &a << std::endl;
	#pragma omp parallel num_threads(8)
	{
		int b;
		#pragma omp critical
		std::cout << omp_get_thread_num() << ":" << &a << " "<< &b << std::endl;
		}
	std::cin.get();
	return 0;
}
```

- reduction用来归约的实现，也就是将训练中的全部求和的值在求和的结果返回！除了“+”归约，/, |, &&等都可以作为归约操作的算法。

```cpp
int sum = 0;
std::cout << omp_get_thread_num() << ":" << & num << std::endl;
#pragma omp parallel num_threads(8) reduction(+:sum)
{
	#pragma omp critical
	std::cout << omp_get_thread_num() << ":" << &num << std::endl;
	#pragma omp for 
	for(int i=1; i<=1000;i++){
		sum+= i;
	}
} 


std::cout << sum << std::endl;

```

 **copyin**： clause让 thread private声明的变量的值和主线程的值相同，如下例子：

```cpp
#include <omp.h>
#include <iostream>
int a ;
#pragma omp threadprivate(a)
int main()
{
    a = 99;
    std::cout << omp_gte_thread_num() << ":" << &a << std::endl;
    #pragma omp parallel num_thread(8) copyin(a)
    { 
        #pragma omp critical 
        std::cout << omp_get_thread_num() << &a << std::endl;
    }
    std::cin.get();
    return 0;
}
```


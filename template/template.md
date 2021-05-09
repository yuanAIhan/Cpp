**C++中的模板：**

https://www.cnblogs.com/liangliangh/p/4219879.html

在模板类的内容很多的时候，可以优先进行一定的初始化，然后来进行对应的结果的代码的验证，反正在出现错误的时候一次性出现很多的错误无法确认问题出现的位置！

  **函数模板**：是一些被参数化的函数，代表的是一个函数家族。可以用多种不同的数据类型进行调用，其外表看起来和普通的函数很相似的，唯一的区别在于就是有些函数元素是没有确定的。下面是一些简答的例子：

```cpp
template <typename T>
inline T const& max (T const& a, T const& b)
{
  return a < b ? b : a;
}
```

  在上面的程序中类型参数是T，可以使用任何标识符作为类型参数的名称，但是一般都是使用T，类型参数可以是任何的类型，但是其中的对应的数据类型必须支持：```operator<``` 才可以。

  同时还可以使用关键字```class```取代其中的```typename```，但是为了不带来迷惑，我个人建议还是使用```typename```的方式更好的！使用```class```只是历史原因，所以一般还是不要用，否则会让人觉得可以使用```struct```的值来取代其中的```typename```迷惑性很强！

```cpp
int main()
{
  int a = ::max(7, 2);
  double b = ::max(8.9, 9.2);
}
```

  每次调用之前都是一个域限定符```::```，因为在```std```中也有一个max的函数，一般指代明确一些会比较好！

  **实例化**：即使用具体的数据类型代替模板参数的过程，会生成一个模板的实列。

```cpp
inline int const& max (int const& a, int const& b)
{
  return a < b ? b : a;
}
```

**实参演绎**：在实参的演绎中是不允许出现自动类型转换，只要是同一个T申明的变量对应的每个T都必须相同才可以！

**模板参数**：

  模板参数：位于函数模板名称前面，在一对<>中进行声明的：```template<typename T>```

  调用参数：位于函数模板名称之后，在一对圆括号内部进行声明。

  ```...max(T const& a, T const& b)```//其中的a 跟 b都是调用参数！

只要是你声明了不同的数据类型作为对应的函数的参数的数据类型的时候，对应的必须在实例化的时候指定对应的满足对应的特定的数据类型的函数参数的值！如果你在对应的函数返回的参数中使用了模板参数的时候，那么必须在实例化的时候进行指定，因为模板参数的演绎是无法得到对应的返回参数的类型。

**重载函数模板**：

```cpp
inline int const& max(int const& a, int const& b)
{
    return a < b ? a : b;
}
template <typename T>
inline T const& max(T const& a, T const& b)
{
    return a < b ? a : b;
}
template <typename T>
inline T const& max(T const& a, T const& b, T const& c)
{
    return ::max(c, ::max(a, b));
}
```

**总结：**

1. ​	模板函数为不同的模板实参定义了一个函数家族
2. ​    当你传递模板实参得到时候，可以根据实参的类型来对函数模板进行实例化
3. ​    可以显示的指定模板的参数
4. ​    可以重载函数模板
5. ​    当重载函数模板的时候，把对应的改变限制在显示的指定模板参数
6. ​    一定要让函数模板的所有重载版本的声明都位于他们的被调用的位置之前的



C++模板的语法：

函数模板和类模板的简单例子：

```cpp
#include <iostream>

// 函数模板
template<typename T>
bool equivalent(const T& a, const T& b){
    return !(a < b) && !(b < a);
}
// 类模板
template<typename T=int> // 默认参数
class bignumber{
    T _v;
public:
    bignumber(T a) : _v(a) { }
    inline bool operator<(const bignumber& b) const; // 等价于 (const bignumber<T>& b)
};
// 在类模板外实现成员函数
template<typename T>
bool bignumber<T>::operator<(const bignumber& b) const{
    return _v < b._v;
}

int main()
{
    bignumber<> a(1), b(1); // 使用默认参数，"<>"不能省略
    std::cout << equivalent(a, b) << '\n'; // 函数模板参数自动推导
    std::cout << equivalent<double>(1, 2) << '\n';
    std::cin.get();    return 0;
}
```

- 类型参数（type template parameter），用 typename 或 class 标记；
- 非类型参数（non-type template parameter）可以是：整数及枚举类型、对象或函数的指针、对象或函数的引用、对象的成员指针，非类型参数是模板实例的常量；
- 模板型参数（template template parameter），如“template<typename T, template<typename> class A> someclass {};”；
- 模板参数可以有默认值（函数模板参数默认是从 C++11 开始支持）；
- 函数模板的和函数参数类型有关的模板参数可以自动推导，类模板参数不存在推导机制；
- C++11 引入变长模板参数，请见下文。也就是参数的个数可以随意改变决定的！

模板特例化：所谓模板特例化即对于通例中的某种或某些情况做单独专门实现，最简单的情况是对每个模板参数指定一个具体值，这成为完全特例化（full specialization），另外，可以限制模板参数在一个范围取值或满足一定关系等，这称为部分特例化（partial specialization），用数学上集合的概念，通例模板参数所有可取的值组合构成全集U，完全特例化对U中某个元素进行专门定义，部分特例化对U的某个真子集进行专门定义。

```cpp
// 实现一个向量类
template<typename T, int N>
class Vec{
    T _v[N];
    // ... // 模板通例（primary template），具体实现
};
template<>
class Vec<float, 4>{
    float _v[4];
    // ... // 对 Vec<float, 4> 进行专门实现，如利用向量指令进行加速
};
template<int N>
class Vec<bool, N>{
    char _v[(N+sizeof(char)-1)/sizeof(char)];
    // ... // 对 Vec<bool, N> 进行专门实现，如用一个比特位表示一个bool
};
```

模板：这里的判断对应的传入的两个类型是否等价的，也就是根据传入到其中的类型的不同来决定对应的传入的程序的类型是否等价的代码实现！

```cpp
template<typename T, int i> class cp00; // 用于模板型模板参数
// 通例
template<typename T1, typename T2, int i, template<typename, int> class CP>
class TMP;
// 完全特例化
template<>
class TMP<int, float, 2, cp00>;
// 第一个参数有const修饰
template<typename T1, typename T2, int i, template<typename, int> class CP>
class TMP<const T1, T2, i, CP>;
// 第一二个参数为cp00的实例且满足一定关系，第四个参数为cp00
template<typename T, int i>
class TMP<cp00<T, i>, cp00<T, i+10>, i, cp00>;
// 编译错误!，第四个参数类型和通例类型不一致
//template<template<int i> CP>
//class TMP<int, float, 10, CP>;

```

- 在定义模板特例之前必须已经有模板通例（primary template）的声明；
- 模板特例并不要求一定与通例有相同的接口，但为了方便使用（体会特例的语义）一般都相同；
- 匹配规则，在模板实例化时如果有模板通例、特例加起来多个模板版本可以匹配，则依据如下规则：对版本AB，如果 A 的模板参数取值集合是B的真子集，则优先匹配 A，如果 AB 的模板参数取值集合是“交叉”关系（AB 交集不为空，且不为包含关系），则发生编译错误，对于函数模板，用函数重载分辨（overload resolution）规则和上述规则结合并优先匹配非模板函数。

模板类实现递归的例子：

```cpp
#include <iostream>
template <int N>
class Tmp{
    public:
    	enum{
            ret = N ==0 ? 1 : N * Tmp<N-1>::ret;
        };
};
int main()
{
    std::cout << Tmp<10>::ret << "\n";
    std::cin.get();
    return 0;
}
//这个代码是存在错误的，因为其中的代码中是不回判断其中的N是否等于0的，其对应的会一直执行的模板的N-1
```

正确的代码：

```cpp
#include <iostream>

// 计算 N 的阶乘 N!
template<int N>
class aTMP{
public:
    enum { ret = N * aTMP<N-1>::ret };
};
template<>
class aTMP<0>{
public:
    enum { ret = 1 };
};

int main(){
    std::cout << aTMP<10>::ret << '\n';
    std::cin.get(); return 0;
}
```



模板元编程：从**编程形式**来看，模板的“<>”中的模板参数相当于函数调用的输入参数，模板中的 typedef 或 static const 或 enum 定义函数返回值（类型或数值，数值仅支持整型，如果需要可以通过编码计算浮点数），代码计算是通过类型计算进而选择类型的函数实现的（C++ 属于静态类型语言，编译器对类型的操控能力很强）。代码示意如下：

可以在编译期完成很多的对应的类型的判断的操作实现！

而且完全可以在对应的模板中实现一个对应的类来作为对应的参数传递进去，因为其中的参数判断的时候肯定是对应的传递来的参数的值有具体的类的方法的时候对应的函数调用才会成功执行的。

也就是说可以根据对应的类型的参数来决定使用什么类型！

```cpp
#include <iostream>
template <typename T, int i = 1>
class someComputing{
    public:
    	using resType = volatile T*;
    	enum {
            retValue = i + someComputing<T, i-1>::retValue
        };
    	static void f()
        {
            std::cout <<"i=" << i << "\n";
        }
}
template<typename T> // 模板特例，递归终止条件
class someComputing<T, 0> {
public:
    enum { retValume = 0 };
};

template<typename T>
class codeComputing {
public:
    static void f() { T::f(); } // 根据类型调用函数，代码计算
};

int main(){
    someComputing<int>::retType a=0;
    std::cout << sizeof(a) << '\n'; // 64-bit 程序指针
    // VS2013 默认最大递归深度500，GCC4.8 默认最大递归深度900（-ftemplate-depth=n）
    std::cout << someComputing<int, 500>::retValume << '\n'; // 1+2+...+500
    codeComputing<someComputing<int, 99>>::f();
    std::cin.get(); return 0;
}
```

```cpp
#include <iostream>

template<int N>
class sumt{
public: static const int ret = sumt<N-1>::ret + N;
};
template<>
class sumt<0>{
public: static const int ret = 0;
};

int main() {
    std::cout << sumt<5>::ret << '\n';
    std::cin.get(); return 0;
}
```

  当编译器遇到 sumt<5> 时，试图实例化之，sumt<5> 引用了 sumt<5-1> 即 sumt<4>，试图实例化 sumt<4>，以此类推，直到 sumt<0>，sumt<0> 匹配模板特例，sumt<0>::ret 为 0，sumt<1>::ret 为 sumt<0>::ret+1 为 1，以此类推，sumt<5>::ret 为 15。值得一提的是，虽然对用户来说程序只是输出了一个编译期常量 sumt<5>::ret，但在背后，编译器其实至少处理了 sumt<0> 到 sumt<5> 共 6 个类型。

​    从这个例子我们也可以窥探 C++ 模板元编程的函数式编程范型，对比结构化求和程序：for(i=0,sum=0; i<=N; ++i) sum+=i; 用逐步改变存储（即变量 sum）的方式来对计算过程进行编程，模板元程序没有可变的存储（都是编译期常量，是不可变的变量），要表达求和过程就要用很多个常量：sumt<0>::ret，sumt<1>::ret，...，sumt<5>::ret 。函数式编程看上去似乎效率低下（因为它和数学接近，而不是和硬件工作方式接近），但有自己的优势：描述问题更加简洁清晰（前提是熟悉这种方式），没有可变的变量就没有数据依赖，方便进行并行化。

```cpp
#include <iostream>
template<typename T>
bool equivalent(const T& a, const T& b)
{
    return !(a<b)&&!(b <a);
}

template<typename T =int>
class bigNumber{
  T _v;
  public:
    bigNumber(T _a) : _v(_a) {}
    inline bool operator<(const bognumber& b) const;
};

template<typename T>
bool bignumber<T>::operator<(const bignumber& b) const{
    return _v < b._v;
}
int main()
{
}
```

**重要规则：模板类中的成员只有被引用的时候才会进行实例化的，这种被称为推迟实例化**

```cpp
#include <iostream>

template<typename T>
class aTMP {
public:
    void f1() { std::cout << "f1()\n"; }
    void f2() { std::ccccout << "f2()\n"; } // 敲错键盘了，语义错误：没有 std::ccccout
};

int main(){
    aTMP<int> a;
    a.f1();
    // a.f2(); // 这句代码被注释时，aTMP<int>::f2() 不被实例化，从而上面的错误被掩盖!
    std::cin.get(); return 0;
}
```

上面的代码中其中调用f2()的方法的时候是不会报错！就是因为其是没有被实例化的！

- 对于一些递归定义的模板的时候，对应的实现的时候必须有一个模板终止的模板特列存在否则对应的代码将会出现错误，因为其是无法判断当前的值是否已经满足对应的条件的！
- 模板的实现的时候应该是从里面的函数往外面看，应该是从内部往外部递归去查看对应的代码！
- 编译器数值计算的强大能力，是可以很好的使用起来的！


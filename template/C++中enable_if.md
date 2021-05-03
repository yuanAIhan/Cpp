**C++元编程初探**：enable_if

  只有传递的第一个参数为true的时候，对应的后面type的定义才有对应的type的值！

```c++
template <bool con, class T = void>
struct enable_if;
```

   如果满足条件则启用类型，如果Cond为true，则将类型T启用为成员类型enable_if :: type。否则，未定义enable_if :: type。这在不满足特定条件时在编译时隐藏签名很有用，因为在这种情况下，将不会定义成员enable_if :: type，并且尝试使用它进行编译将失败。它的定义行为等同于：

```cpp
template<bool Cond, class T = void> 
struct enable_if {
};

template<class T> 
struct enable_if<true, T> { 
	typedef T type; 
};
```

**用法一：检验函数模板参数类型**:有时候定义的模板函数，只希望特定的类型才可以调用。

```cpp
// 1. 返回类型（布尔）仅在T为整数类型时有效：
template <class T>
typename std::enable_if<std::is_integral<T>::value,bool>::type
is_odd (T i) {
    return bool(i%2);
}

// 2. 第二个模板参数仅在T为整数类型时有效：
template < class T,
           class = typename std::enable_if<std::is_integral<T>::value>::type>
bool is_even (T i) {
    return !bool(i%2);
}

int main() {
  int i = 1;    // 如果i的类型不是整数，则代码无法编译
  std::cout << "i is odd: " << is_odd(i) << std::endl;
  std::cout << "i is even: " << is_even(i) << std::endl;

  return 0;
}
```

在kokkos中主要使用的就是涉及判断前者为真的时候，后者才有使用的价值的也就是对应的才取后者的值为对应的结果类型的实现：

```cpp
  explicit inline View( 
      const Impl::ViewCtorProp<P...>& arg_prop,
      typename std::enable_if<!Impl::ViewCtorProp<P...>::has_pointer,
                              typename traits::array_layout>::type const&
          arg_layout)  //只有第二个值为真的时候对应的才可以得到第二个值得结果！
      : m_track(), m_map()
      {}

```

上面的代码就代表着，只有前面的结果为真的时候，对应的值得到的类型的值才会是array_layout的值。
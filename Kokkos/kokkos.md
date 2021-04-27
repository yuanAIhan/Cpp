**kokkos中的一些实现细节！**主要是计算特性一块的复现上！

  类:Dimension:首先dimension中定义了两种维度，然后不同的维度也就是是否在编译的时候指定，同时使用的参数标记分别为:rank 和 dynamic_rank的值，实现的时候如果对应的维度的大小值为：*KOKKOS_INVALID_INDEX*则说明没有那么多维度，不需要8维那么多。当对应的维度的值设置为1的时候，对应则是忽略这一个维度的大小值。然后当某一个维度的值为 0 的时候对应的就代表着这是一个运行时候的维度，0 的时候代表的是运行时维度， 1 代表的时候没有设定对应的维度，其余的则是编译时候的维度值。

```cpp
enum { ArgN##R = (V != KOKKOS_INVALID_INDEX ? V : 1) };                 
enum { N##R = (V != KOKKOS_INVALID_INDEX ? V : 1) };    
```

  记住很多都是通过继承的方式将一个类中的全部属性都在另外的一个类中可以使用，当然这里的继承必须是按照public的继承方式来实现的，同时也有一些实现的时候是按照将其中的值传递到模板类中实现的！

```cpp
enum { rank = sizeof...(Vals) }; //总维度数
enum { rank_dynamic = Impl::rank_dynamic<Vals...>::value }; //运行时维度数
```

```cpp
template <size_t N> //prepend 在前面添加一个维度
struct prepend {
    typedef ViewDimension<N, Vals...> type;
};


template <size_t N> //append 在后面添加一个维度
struct append {
    typedef ViewDimension<Vals..., N> type;
};


template <class A, class B>
struct ViewDimensionJoin;

template <size_t... A, size_t... B> //合并维度（运行时维度+编译时维度）
struct ViewDimensionJoin<ViewDimension<A...>, ViewDimension<B...>> {
  typedef ViewDimension<A..., B...> type;
};

size_t extent(const unsigned r) const noexcept {}//获取某一维度的大小
static KOKKOS_INLINE_FUNCTION constexpr size_t static_extent( }
    //静态函数，获取某一编译时维度的大小，若返回值为0，说明该维度为运行时维度
```

  类：ViewOffset：这个文件应该主要是用来解决对应的针对于不同的偏移的时候其中的内存访问和对应的地址的获取的操作的函数，看其中的一个函数对应的维度的值的访问的操作！

rank 3 可以这样理解：把前面两维的结果合并成一维，然后相当于调用operator(I0,I1),合并时，得到的结果为 i1 + m_dim.N1 \* i0，再把这个看作I0，再调用operator(I0,I1)，*即调用operator(i1 + m_dim.N1 \* i0,i2)，所以得到 i2 + m_dim.N2 \* (i1 + m_dim.N1 \* (i0))* 

也可以这样理解：二维平面上的每一个点是一个一维数组，第三维是指在这个数组中的位置，数组的大小就是m_dim.N2*

```cpp
template <typename I0, typename I1, typename I2>
KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const& i0,
                                                      I1 const& i1,
                                                      I2 const& i2) const {
    return i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0));
}
```

 layout()函数返回view的布局信息，dimension_i()的函数是返回第i维的大小（维度从0开始）

```cpp
constexpr array_layout layout() const {
return array_layout(m_dim.N0, m_dim.N1, m_dim.N2, m_dim.N3, m_dim.N4,
	m_dim.N5, m_dim.N6, m_dim.N7);
}

KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
}
```

*size()函数返回view的大小(各个维度的乘积)* 和 span的方式几乎完全相同！

```cpp
constexpr size_type size() const {
	return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
m_dim.N6 * m_dim.N7;
}
```

*stride_i()函数返回该维度的两个相邻下标的步长*

```cpp
KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return 1; }
KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
	return m_dim.N7;
}
```

*  获取view各个维度的步长大小，存储到数组s中，参数是s一个指针(指向一个大小为8的一维数组)*，其实就是将上面的函数的每一个维度的值都求解出来然后用于此处的操作实现！

```cpp
template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const { 
    size_type n = 1;
    if (7 < dimension_type::rank) {
      s[7] = n;
      n *= m_dim.N7;
    }
    if (6 < dimension_type::rank) {
      s[6] = n;
      n *= m_dim.N6;
    }
    if (5 < dimension_type::rank) {
      s[5] = n;
      n *= m_dim.N5;
    }
    if (4 < dimension_type::rank) {
      s[4] = n;
      n *= m_dim.N4;
    }
    if (3 < dimension_type::rank) {
      s[3] = n;
      n *= m_dim.N3;
    }
    if (2 < dimension_type::rank) {
      s[2] = n;
      n *= m_dim.N2;
    }
    if (1 < dimension_type::rank) {
      s[1] = n;
      n *= m_dim.N1;
    }
    if (0 < dimension_type::rank) {
      s[0] = n;
    }
    s[dimension_type::rank] = n * m_dim.N0;
}
```

下面是定义了对应的三个构造函数，都是拷贝构造函数，但是传递的参数的布局方式是不同的，在其中将对应的三种的布局方式全部都给出来啦，其中一个是对应的为right的构造函数，另外一个是对应的left的构造函数，最后一个是对应的为layoutStride的布局方式！

```cpp
  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset( 
      const ViewOffset<DimRHS, Kokkos::LayoutRight, void>& rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }
  
```

对应的实现的不同的代码，主要是针对于不同的布局的方式的一个定义，而且是完全可以按照用户自己实现的定义的布局方式来实现代码的书写的，但是感觉这个功能没有太大的必要！

```cpp
template <>
struct ViewStride<8> {
  size_t S0, S1, S2, S3, S4, S5, S6, S7;

  ViewStride()                  = default;
  ViewStride(const ViewStride&) = default;
  ViewStride& operator=(const ViewStride&) = default;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3,
                       size_t aS4, size_t aS5, size_t aS6, size_t aS7)
      : S0(aS0),
        S1(aS1),
        S2(aS2),
        S3(aS3),
        S4(aS4),
        S5(aS5),
        S6(aS6),
        S7(aS7) {}
};

template <class Dimension>
struct ViewOffset<Dimension, Kokkos::LayoutStride, void> {
 private:
  using stride_type = ViewStride<Dimension::rank>;
};
```

**ViewArrayAnalysis类**中一些方法的学习，因为对应的方法中主要涉及有一些递归调用的方式将其中的对应的值为传入的维度的值进行从右到左的一个递归调用的实现。其借用了之前的函数方式将其中的维度的值进行在后面的维度拼接的方式！通过局部特化出ViewArrayAnalysis<T[N]>，ViewArrayAnalysis<T[]>和ViewArrayAnalysis<T\*>，然后在内部再调用ViewArrayAnalysis<T>，形成递归，把Data type(多维数组类型)({const} value_type ***[#][#][#])从右往左不断地降维，最后得到数组的数据类型value_type*。

这个类就是在上面的代码的基础上，将前面的对应的类型的代码进行一个得到其中的维度的值的一个处理，同时从中可以得到不同的值的方法实现。就是为了将其中的维度的值得到后存储在对应的：rank 和 rank_dynamic之中的。

**类ViewValueFunctor**的实现，其主要是用来将对应的构造函数、对齐、析构整合到一个函数子类之中去的。

```cpp
template <class ExecSpace, class ValueType,
          bool IsScalar = std::is_scalar<ValueType>::value> 
          //三个模板参数：ExecSpace，ValueType，IsScalar
struct ViewValueFunctor;
```

**ViewdataHandler的类**，其中主要是一些基本数据类型的定义实现上，然后在后面完成对应的参数通过给出的下标参数，调用offset计算出该下标元素的地址偏移值(相对于第一个元素的偏移值)*的方式实现，感觉总的来说类似是完成其中[]访问元素的下标的方式重载实现。其对应得跟前面的重载了operator()的操作是一样，其中的操作的时候对应的都是使用其中的[]的方式有完成一个重载来访问其中的数据类型的引用的，因为mapping中虽然实现了但是因为必须在View中继续实现其才是可以完成在view中使用的！然而其中的代码量就是因为这些部分的存在导致对应的代码量非常的距大！

View中：

首先其中实现的函数access的方式，其的实现是对应的跟其中的operator()和[]和access()的方式是相似的。所以对应的在处理的时候应该多查看其中的各个代码的具体功能，其实其中的很多代码是冗余的。

```cpp
template <typename I0, typename I1, typename I2, typename I3, typename I4,
	typename I5, class... Args>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
(Kokkos::Impl::are_integral<I0, I1, I2, I3, I4, I5, Args...>::value &&
(6 == Rank) && !is_default_map),
reference_type>::type
	access(const I0& i0, const I1& i1, const I2& i2, const I3& i3, const I4& i4,
const I5& i5, Args... KOKKOS_IMPL_SINK(args)) const {
KOKKOS_IMPL_VIEW_OPERATOR_VERIFY(
KOKKOS_IMPL_SINK((m_track, m_map, i0, i1, i2, i3, i4, i5, args...)))
	return m_map.reference(i0, i1, i2, i3, i4, i5);
}
```

除了很多相互联系的类外，对应还有一些函数的使用，因为还需要完成对应的函数的处理！首先需要获取得到的参数为需要分配的内存空间的大小：

```cpp
  static constexpr size_t required_allocation_size(
      const size_t arg_N0 = 0, const size_t arg_N1 = 0, const size_t arg_N2 = 0,
      const size_t arg_N3 = 0, const size_t arg_N4 = 0, const size_t arg_N5 = 0,
      const size_t arg_N6 = 0, const size_t arg_N7 = 0) {
    return map_type::memory_span(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }
```

**总的来说的话**，就是对应的因为viewdimension的设置，然后出现后面的不同的一些属性对于其的处理实现，








```cpp
using view_type = Kokkos::View<double ***[][]> ;


using View_type = Kokkos::view<double, void*, void*,void*, void*>;



```



```cpp
 *   - View< DataType >
 *   - View< DataType , Layout >
 *   - View< DataType , Layout , Space >
 *   - View< DataType , Layout , Space , MemoryTraits >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
```



```cpp
using view_type = Kokkos::view<double *, Layout::layoutleft>;

using View_type = Kokkos::view<double, void*, void*,void*, void*, Layout::layoutleft, Space, MemoryTraits>;
//

```



首先还是先实现一个代码，然后再增加对应的一些功能，功能上主要是完成对应的layout之后，然后再去实现后者的memory的。
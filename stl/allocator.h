#ifndef _JJALLOC_
#define _JJALLOC_

#include <iostream>
#include <malloc.h>
#include <new>
#include <cstddef>
#include <cstdlib>
#include <climits>

namespace JJ
{
  template <class T>
  inline T* _allocate(std::ptrdiff_t size, T*) 
  {
    T* temp = (T*)(::operator new((size_t)(size * sizeof(T*))));
    if(temp == 0 ) {
      std::cerr << "out of memory" << std::endl;
      exit(1);
    }
    return temp;
  }
  template <class T>
  inline void _deallocate(T* buffer) {
    ::operator delete(buffer);
  }
  template <class T1, class T2>
  inline void _construct(T1* p, const T2& value) {
    new(p) T1(value);
  }
  template <class T>
  inline void _destory(T* ptr) {
    ptr -> ~T();
  }

    template <class T>
    class allocator{
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;

        template <class U>
        struct rebind{
            typedef allocator<U> other;
        };
        pointer allocate(size_type n, const void* hint = 0 ) {
            return _allocate((difference_type)n, (pointer)0);
        }
        void deallocate(pointer p, size_type n) {
            _deallocate(p);
        }
        void construct(pointer p, const T& value) {
            _construct(p, value);
        }
        void destory(pointer p) {
            _deallocate(p);
        }
        pointer address(reference x) {
            return (pointer)x;
        }
        const_pointer const_address(const_reference x) {
            return (const_pointer)&x;
        }
        size_type max_size() const{
            return size_type(UINT_MAX / sizeof(T));
        }
    };
}
#endif

template<class T, class... Dim> 
struct vector;

template<class T>
struct vector<T, JJ::allocator<T>>{

};

template<typename T1, template<typename>class Cont> 
struct node{

};
//注意这两种语法的差别

namespace JJ{
  #if 0
    #include <new>
    #define __THROW_BAD_ALLOC throw bad_alloc
  #elif !defined(__THROW_BAND_ALLOC)
    #include <iostream>
    #define __THROW_BAD_ALLOC std::cerr << "out of memory" << std::endl; exit(0)
  #endif
  

  template <int inst>
  class __malloc_alloc_template {
    static void *oom_malloc(size_t) ;
    static void *oom_realloc(void*, size_t);
    static void (* __malloc_alloc_oom_handler)();

    public:
      static void * allocate (size_t n)
      {
        void *result = malloc(n);
        if(0 == result ) result = oom_malloc(n);
        return result;
      }
      static void deallocate(void *p, size_t ) {
        free(p);
      }
      static void* realloctae(void *p, size_t , size_t new_size) 
      {
        void *result = realloc(p, new_size);
        if(0 == result ) result = oom_realloc(p, new_size);
        return result;
      }
      static void (*set_malloc_handler(void (*f)()))()
      {
        void(*old) () = __malloc_alloc_oom_handler;
        __malloc_alloc_oom_handler = f;
        return(old);
      }
  };
  template <int inst>
  void (* __malloc_alloc_template<inst>::__malloc_alloc_oom_handler) () = 0;

  template <int inst>
  void * __malloc_alloc_template<inst>::oom_malloc(size_t n)
  {
    void (*my_alloc_handler)();
    void *result;
    for(;;) {
      my_alloc_handler = __malloc_alloc_oom_handler;
      if(0 == my_alloc_handler) {
        __THROW_BAD_ALLOC;
      }
      (*my_alloc_handler)();
      result = malloc(n);
      if(result) return (result);
    }
  }

  template <int inst>
  void * __malloc_alloc_template<inst>::oom_realloc(void*p, size_t n)
  {
    void (*my_alloc_handler)();
    void *result;
    for(;;) {
      my_alloc_handler = __malloc_alloc_oom_handler;
      if( 0 == my_alloc_handler) {
        __THROW_BAD_ALLOC;
      }
      (*my_alloc_handler)();
      result = realloc(p, n);
      if(result) return (result);
    }
  }
}
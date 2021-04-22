#include <vcruntime.h>
enum {__ALIGN = 8};
enum {__MAX_BYTES = 128};
enum {__NFREELISITS = __MAX_BYTES / __ALIGN};


union obj {
  union obj * free_list_link;
  char client_data[1];
};
///类似于结构体的定义实现！

template <bool threads, int inst>
class __default_alloc_template {
  private:
    static size_t ROUNF_UP(size_t bytes){
      return (((bytes) + __ALIGN - 1) & ~(__ALIGN - 1));
    }
  private:
    union obj{
      union obj * free_list_link;
      char client_data[1];
    };
  private:
    static obj * volatile free_list[__NFREELISITS];
    static size_t FREELIST_INDEX(size_t bytes) {
      return (((bytes) + __ALIGN - 1) / __ALIGN - 1 );
    }
  public:
    static void * allocate(size_t n){}
    static void * deallocate(void *p, size_t n){}
    static void * reallocate(void *p, size_t old_sz, size_t new_sz){}
    
};

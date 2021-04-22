#include <vcruntime.h>
#include <vector>
template <class T, class Alloc = alloc>
class vector {
  public:
    using value_type = T;
    using iterator = value_type*;
    using size_type = size_t;
  protected:
    iterator start;
    iterator finish;
    iterator end_of_storage;
  public:
    iterator begin() {return start;}
    iterator end() {return finish;}
    size_type size() {return size_type(end() - begin());}
    size_type capacity() const{
      return size_type(end_of_storage - begin());
    }
    
};
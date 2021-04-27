#include <vcruntime.h>
#include <vector>
template <class T, class Alloc = alloc>
class vector {
  public:
    using value_type = T;
    using iterator = value_type*;
    using size_type = size_t;
    using reference = value_type&;
    using pointer = value_type*;

  protected:
    iterator start;
    iterator finish;
    iterator end_of_storage;
    void insert_aux(iterator position, const T& x);
    void deallocate() {
        if(start) {
            data_allocate::deallocate(start, end_of_storage - start);
        }
    } 
  public:
    iterator begin() {return start;}
    iterator end() {return finish;}
    size_type size() {return size_type(end() - begin());}
    size_type capacity() const{
      return size_type(end_of_storage - begin());
    }

};
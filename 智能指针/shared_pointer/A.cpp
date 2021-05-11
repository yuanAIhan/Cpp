#include <stdlib.h>

//类似于shared Alloc.hpp中的文件的代码
class A{
protected:
    int alloc_size;
    unsigned int* use = nullptr;
public:
    A():use(nullptr){}

    A(int number) 
    {
        alloc_size = number;
        use = new unsigned int(1);
    }
    
    void increment(A* record)
    {
        record->use ++;
    }
    void decrement(A* record)
    {
        record->use --;
        if( record->use == 0) 
        {
            destory(record);
        }
    }
    void destory(A* record)
    {
        record = nullptr;
    }
    int get_use()
    {
        return *use;
    }
};
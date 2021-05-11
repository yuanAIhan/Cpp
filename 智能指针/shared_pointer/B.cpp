#include "A.cpp"
#include <iostream>

class B : public A
{
        int b;
    public:
        using record = A; //RecordBase;
        B(): b(0){};
        
};
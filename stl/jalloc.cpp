#include "allocator.h"
#include <vector>
#include <iostream>
#include "Kokkos_SharedAlloc.hpp"
#include "Kokkos_SharedAlloc.cpp"

using namespace std;
class getnum{
    
};

int main()
{

    int ia[5] = {0, 1, 2, 3, 4};
    for(int i = 0; i < 5; i ++) {
        cout << &ia[i] << " ";
    }
    cout << endl;
    unsigned int i;
    vector<int, JJ::allocator<int> > iv(ia, ia + 5);
    for(int i = 0; i < iv.size(); i ++) {
        cout << iv[i] << " ";
    }
    cout << endl;
    for(int i = 0; i < iv.size(); i ++) {
        cout << &iv[i] << " ";
    }
    cout << endl;

    return 0;
}
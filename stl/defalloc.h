#ifndef DEFALLOC_H
#define DEFALLOC_H

#include <new.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <vcruntime.h>
#include <memory>
// #include <algobase.h>

template <class T>
inline T* allocate(ptrdiff_t size, T*) {
  
}
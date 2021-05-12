#include <stdio.h>

int nElem = 6;

dim3 block(3);
dim3 grid((nElem + block.x - 1) / block.x);

int main(void)
{
    printf("grid.x %d  gird.y  %d  grid.z %d \n", grid.x, grid.y, grid.z);
    printf("block.x %d  block.y  %d  block.z %d \n", block.x, block.y, block.z);
    return 0;
}
#include <stdio.h>
int main()
{
    int count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&count)){return -1;}
    if (count == 0) {return -1;}
    for (int device = 0; device < count; ++device)
    {
        cudaDeviceProp prop;
        if (cudaSuccess != cudaGetDeviceProperties(&prop, device)){ continue;}
        printf("%d.%d ", prop.major, prop.minor);
    }
    return 0;
}

#include<dawn/webgpu_cpp.h>

int main(int /*argc*/, char** /*argv*/)
{
    wgpu::Device device1;
    device1.Release();
    return 0;
}
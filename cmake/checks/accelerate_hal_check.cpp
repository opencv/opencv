#include <Accelerate/Accelerate.h>

int main()
{
    vImage_Buffer src = {}, dst = {};
    float kernel = 1.f;
    return (int)vImageSepConvolve_PlanarF(&src, &dst, nullptr, 0, 0, &kernel, 1, &kernel, 1, 0.f, 0.f, kvImageNoFlags);
}

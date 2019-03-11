#include <VX/vx.h>
int main()
{
    return VX_REFERENCE_COUNT == VX_REFERENCE_TYPE ? VX_REFERENCE_NAME : 0;
}

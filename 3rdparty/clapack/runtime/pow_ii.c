#include "f2c.h"

int pow_ii(int *ap, int *bp)
{
    int p;
    int x = *ap;
    int n = *bp;

    if (n <= 0) {
        if (n == 0 || x == 1)
            return 1;
        return x != -1 ? 0 : (n & 1) ? -1 : 1;
    }
    unsigned u = (unsigned)n;
    for(p = 1; ; )
    {
        if(u & 01)
            p *= x;
        if(u >>= 1)
            x *= x;
        else
            break;
    }
    return p;
}

#include "clapack.h"

integer pow_ii(integer *ap, integer *bp)
{
    integer pow, x, n;
    unsigned long u;

    x = *ap;
    n = *bp;

    if (n <= 0) {
        if (n == 0 || x == 1)
            return 1;
        return x != -1 ? 0 : (n & 1) ? -1 : 1;
    }
    u = n;
    for(pow = 1; ; )
    {
        if(u & 01)
            pow *= x;
        if(u >>= 1)
            x *= x;
        else
            break;
    }
    return(pow);
}

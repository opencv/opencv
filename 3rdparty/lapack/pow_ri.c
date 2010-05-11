#include "clapack.h"

double pow_ri(real *ap, integer *bp)
{
    double pow, x;
    integer n;
    unsigned long u;

    pow = 1;
    x = *ap;
    n = *bp;

    if(n != 0)
    {
        if(n < 0)
        {
            n = -n;
            x = 1/x;
        }
        for(u = n; ; )
        {
            if(u & 01)
                pow *= x;
            if(u >>= 1)
                x *= x;
            else
                break;
        }
    }
    return(pow);
}

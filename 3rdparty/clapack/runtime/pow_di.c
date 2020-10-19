#include "f2c.h"

double pow_di(double *ap, int *bp)
{
    double p = 1;
    double x = *ap;
    int n = *bp;

    if(n != 0)
    {
        if(n < 0)
        {
            n = -n;
            x = 1/x;
        }
        for(unsigned u = (unsigned)n; ; )
        {
            if(u & 01)
                p *= x;
            if(u >>= 1)
                x *= x;
            else
                break;
        }
    }
    return p;
}

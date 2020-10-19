#include "f2c.h"

void d_cnjg(lapack_doublecomplex *r, lapack_doublecomplex *z)
{
	double zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

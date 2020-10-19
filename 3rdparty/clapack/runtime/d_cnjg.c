#include "f2c.h"

void d_cnjg(doublecomplex *r, doublecomplex *z)
{
	doublereal zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

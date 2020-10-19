#include "f2c.h"

void r_cnjg(complex *r, complex *z)
{
	float zi = z->i;
	r->r = z->r;
	r->i = -zi;
}

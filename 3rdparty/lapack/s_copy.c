/* Unless compiled with -DNO_OVERWRITE, this variant of s_copy allows the
 * target of an assignment to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90),
 * as in  a(2:5) = a(4:7) .
 */

#include "clapack.h"

/* assign strings:  a = b */

void s_copy(register char *a, register char *b, ftnlen la, ftnlen lb)
{
	register char *aend, *bend;

	aend = a + la;

	if(la <= lb)
		if (a <= b || a >= b + la)
			while(a < aend)
				*a++ = *b++;
		else
			for(b += la; a < aend; )
				*--aend = *--b;
	else {
		bend = b + lb;
		if (a <= b || a >= bend)
			while(b < bend)
				*a++ = *b++;
		else {
			a += lb;
			while(b < bend)
				*--a = *--bend;
			a += lb;
			}
		while(a < aend)
			*a++ = ' ';
	}
}

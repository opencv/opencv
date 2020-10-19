/* Unless compiled with -DNO_OVERWRITE, this variant of s_copy allows the
 * target of an assignment to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90),
 * as in  a(2:5) = a(4:7) .
 */

#include "f2c.h"

/* assign strings:  a = b */

int s_copy(char *a, char *b)
{
    strcpy(a, b);
    return 0;
}

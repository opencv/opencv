/* Unless compiled with -DNO_OVERWRITE, this variant of s_cat allows the
 * target of a concatenation to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90).
 */

#include "f2c.h"

int s_cat(char *lp, char **rpp, int* rnp, int *np)
{
    int i, L = 0;
    int n = *np;

    for(i = 0; i < n; i++) {
        int ni = rnp[i];
        if(ni > 0) {
            memcpy(lp + L, rpp[i], ni);
            L += ni;
        }
    }
    lp[L] = '\0';
    return 0;
}

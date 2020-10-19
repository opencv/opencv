/* Unless compiled with -DNO_OVERWRITE, this variant of s_cat allows the
 * target of a concatenation to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90).
 */

#include "f2c.h"

int s_cat(char *lp, char **rpp, ftnint* rnp, ftnint *np)
{
    ftnlen i, L = 0;
    ftnlen n = *np;
    
    for(i = 0; i < n; i++) {
        int ni = rnp[i];
        ni = min(F2C_STR_MAX-1-L, ni);
        if(ni > 0) {
            memcpy(lp + L, rpp[i], ni);
            L += ni;
        }
    }
    lp[L] = '\0';
    return 0;
}

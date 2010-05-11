/* Unless compiled with -DNO_OVERWRITE, this variant of s_cat allows the
 * target of a concatenation to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90).
 */

#include "clapack.h"
#include "stdio.h"
#undef abs
#undef min
#undef max
#include "stdlib.h"
#include "string.h"

void s_cat(char *lp, char *rpp[], ftnint rnp[], ftnint *np, ftnlen ll)
{
    ftnlen i, nc;
    char *rp;
    ftnlen n = *np;
    ftnlen L, m;
    char *lp0, *lp1;

    lp0 = 0;
    lp1 = lp;
    L = ll;
    i = 0;
    while(i < n) {
        rp = rpp[i];
        m = rnp[i++];
        if (rp >= lp1 || rp + m <= lp) {
            if ((L -= m) <= 0) {
                n = i;
                break;
            }
            lp1 += m;
            continue;
        }
        lp0 = lp;
        lp = lp1 = F77_aloc(L = ll, "s_cat");
        break;
    }
    lp1 = lp;
    for(i = 0 ; i < n ; ++i) {
        nc = ll;
        if(rnp[i] < nc)
            nc = rnp[i];
        ll -= nc;
        rp = rpp[i];
        while(--nc >= 0)
            *lp++ = *rp++;
    }
    while(--ll >= 0)
        *lp++ = ' ';
    if (lp0) {
        memcpy(lp0, lp1, L);
        free(lp1);
    }
}

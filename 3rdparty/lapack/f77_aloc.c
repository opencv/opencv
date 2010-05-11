#include "clapack.h"
#undef abs
#undef min
#undef max
#include "stdio.h"

static integer memfailure = 3;

#include "stdlib.h"

char* F77_aloc(integer Len, char *whence)
{
    char *rv;
    unsigned int uLen = (unsigned int) Len;	/* for K&R C */

    if (!(rv = (char*)malloc(uLen))) {
        fprintf(stderr, "malloc(%u) failure in %s\n",
            uLen, whence);
        exit_(&memfailure);
    }
    return rv;
}

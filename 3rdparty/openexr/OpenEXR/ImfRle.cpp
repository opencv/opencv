//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include <string.h>
#include "ImfRle.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

const int MIN_RUN_LENGTH = 3;
const int MAX_RUN_LENGTH = 127;

}

//
// Compress an array of bytes, using run-length encoding,
// and return the length of the compressed data.
//

int
rleCompress (int inLength, const char in[], signed char out[])
{
    const char *inEnd = in + inLength;
    const char *runStart = in;
    const char *runEnd = in + 1;
    signed char *outWrite = out;

    while (runStart < inEnd)
    {
	while (runEnd < inEnd &&
	       *runStart == *runEnd &&
	       runEnd - runStart - 1 < MAX_RUN_LENGTH)
	{
	    ++runEnd;
	}

	if (runEnd - runStart >= MIN_RUN_LENGTH)
	{
	    //
	    // Compressable run
	    //

	    *outWrite++ = (runEnd - runStart) - 1;
	    *outWrite++ = *(signed char *) runStart;
	    runStart = runEnd;
	}
	else
	{
	    //
	    // Uncompressable run
	    //

	    while (runEnd < inEnd &&
		   ((runEnd + 1 >= inEnd ||
		     *runEnd != *(runEnd + 1)) ||
		    (runEnd + 2 >= inEnd ||
		     *(runEnd + 1) != *(runEnd + 2))) &&
		   runEnd - runStart < MAX_RUN_LENGTH)
	    {
		++runEnd;
	    }

	    *outWrite++ = runStart - runEnd;

	    while (runStart < runEnd)
	    {
		*outWrite++ = *(signed char *) (runStart++);
	    }
	}

	++runEnd;
    }

    return outWrite - out;
}


//
// Uncompress an array of bytes compressed with rleCompress().
// Returns the length of the oncompressed data, or 0 if the
// length of the uncompressed data would be more than maxLength.
//

int
rleUncompress (int inLength, int maxLength, const signed char in[], char out[])
{
    char *outStart = out;

    while (inLength > 0)
    {
	if (*in < 0)
	{
	    int count = -((int)*in++);
	    inLength -= count + 1;

	    if (0 > (maxLength -= count))
		return 0;

        // check the input buffer is big enough to contain
        // 'count' bytes of remaining data
        if (inLength < 0)
          return 0;

        memcpy(out, in, count);
        out += count;
        in  += count;
	}
	else
	{
	    int count = *in++;
	    inLength -= 2;

	    if (0 > (maxLength -= count + 1))
		return 0;

        // check the input buffer is big enough to contain
        // byte to be duplicated
        if (inLength < 0)
          return 0;

        memset(out, *(char*)in, count+1);
        out += count+1;

	    in++;
	}
    }

    return out - outStart;
}




OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

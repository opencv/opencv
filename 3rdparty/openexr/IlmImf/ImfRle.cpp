///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

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

        memset(out, *(char*)in, count+1);
        out += count+1;

	    in++;
	}
    }

    return out - outStart;
}




OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2013, Industrial Light & Magic, a division of Lucas
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


#ifndef INCLUDED_IMF_DEEPIMAGESTATE_H
#define INCLUDED_IMF_DEEPIMAGESTATE_H

//-----------------------------------------------------------------------------
//
//      enum DeepImageState -- describes how orderly the pixel data
//      in a deep image are
//
//      The samples in a deep image pixel may be sorted according to
//      depth, and the sample depths or depth ranges may or may not
//      overlap each other.  A pixel is
//
//          - SORTED if for every i and j with i < j
//
//              (Z[i] < Z[j]) || (Z[i] == Z[j] && ZBack[i] < ZBack[j]),
//
//          - NON_OVERLAPPING if for every i and j with i != j
//
//              (Z[i] <  Z[j] && ZBack[i] <= Z[j]) ||
//              (Z[j] <  Z[i] && ZBack[j] <= Z[i]) ||
//              (Z[i] == Z[j] && ZBack[i] <= Z[i] & ZBack[j] > Z[j]) ||
//              (Z[i] == Z[j] && ZBack[j] <= Z[j] & ZBack[i] > Z[i]),
//
//          - TIDY if it is SORTED and NON_OVERLAPPING,
//
//          - MESSY if it is neither SORTED nor NON_OVERLAPPING.
//
//      A deep image is
//
//          - MESSY if at least one of its pixels is MESSY,
//          - SORTED if all of its pixels are SORTED,
//          - NON_OVERLAPPING if all of its pixels are NON_OVERLAPPING,
//          - TIDY if all of its pixels are TIDY.
//
//      Note: the rather complicated definition of NON_OVERLAPPING prohibits
//      overlapping volume samples, coincident point samples and point samples
//      in the middle of a volume sample, but it does allow point samples at
//      the front or back of a volume sample.
//
//-----------------------------------------------------------------------------

#include "ImfNamespace.h"
#include "ImfExport.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

enum DeepImageState
{
    DIS_MESSY = 0,
    DIS_SORTED = 1,
    DIS_NON_OVERLAPPING = 2,
    DIS_TIDY = 3,

    DIS_NUMSTATES   // Number of different image states
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif

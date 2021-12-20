//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

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

#include "ImfExport.h"
#include "ImfNamespace.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

enum IMF_EXPORT_ENUM DeepImageState : int
{
    DIS_MESSY = 0,
    DIS_SORTED = 1,
    DIS_NON_OVERLAPPING = 2,
    DIS_TIDY = 3,

    DIS_NUMSTATES   // Number of different image states
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif

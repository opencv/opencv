//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#include <ImfPartType.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using std::string;

bool isImage(const string& name)
{
    return (name == SCANLINEIMAGE || name == TILEDIMAGE);
}

bool isTiled(const string& name)
{
    return (name == TILEDIMAGE || name == DEEPTILE);
}

bool isDeepData(const string& name)
{
    return (name == DEEPTILE || name == DEEPSCANLINE);
}

bool isSupportedType(const string& name)
{
    return (name == SCANLINEIMAGE || name == TILEDIMAGE ||
            name == DEEPSCANLINE || name == DEEPTILE);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT

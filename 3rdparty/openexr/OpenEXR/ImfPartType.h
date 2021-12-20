//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFPARTTYPE_H_
#define IMFPARTTYPE_H_

#include "ImfExport.h"
#include "ImfNamespace.h"

#include <string>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


static const std::string SCANLINEIMAGE = "scanlineimage";
static const std::string TILEDIMAGE    = "tiledimage";
static const std::string DEEPSCANLINE  = "deepscanline";
static const std::string DEEPTILE      = "deeptile";

IMF_EXPORT bool isImage(const std::string& name);

IMF_EXPORT bool isTiled(const std::string& name);

IMF_EXPORT bool isDeepData(const std::string& name);

IMF_EXPORT bool isSupportedType(const std::string& name);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif /* IMFPARTTYPE_H_ */

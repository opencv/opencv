//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Weta Digital, Ltd and Contributors to the OpenEXR Project.
//



#ifndef INCLUDED_IMF_STRINGVECTOR_ATTRIBUTE_H
#define INCLUDED_IMF_STRINGVECTOR_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class StringVectorAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"

#include <string>
#include <vector>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

typedef std::vector<std::string> StringVector;
typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::StringVector> StringVectorAttribute;

#ifndef COMPILING_IMF_STRING_VECTOR_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::StringVector>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif

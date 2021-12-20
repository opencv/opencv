//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Weta Digital, Ltd and Contributors to the OpenEXR Project.
//



#ifndef INCLUDED_IMF_FLOATVECTOR_ATTRIBUTE_H
#define INCLUDED_IMF_FLOATVECTOR_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class FloatVectorAttribute
//
//-----------------------------------------------------------------------------

#include "ImfExport.h"
#include "ImfNamespace.h"

#include "ImfAttribute.h"

#include <vector>


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

typedef std::vector<float>
    FloatVector;

typedef TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::FloatVector>
    FloatVectorAttribute;

#ifndef COMPILING_IMF_FLOAT_VECTOR_ATTRIBUTE
extern template class IMF_EXPORT_EXTERN_TEMPLATE TypedAttribute<FloatVector>;
#endif


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif

#include "ImfHeader.h"
#include "ImfNamespace.h"
#include "ImfExport.h"

#include <cstddef>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


const unsigned short* get_dwaCompressorNoOp();
const unsigned short* get_dwaCompressorToLinear();
const unsigned short* get_dwaCompressorToNonlinear();

//const unsigned int* get_closestDataOffset();
//const unsigned short* get_closestData();
static inline
const unsigned short* get_dwaClosest(int idx)
{
    throw std::runtime_error("OpenEXR: DW* compression tables are not available");
}

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

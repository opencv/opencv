#include "../precomp.hpp"
#if defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER

#if defined(TH_DISABLE_HEAP_TRACKING)
#elif (defined(__unix) || defined(_WIN32))
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#include "THGeneral.h"

#endif

#ifndef __IPP_HAL_UTILS_HPP__
#define __IPP_HAL_UTILS_HPP__

#include "ippversion.h"
#ifndef IPP_VERSION_UPDATE // prior to 7.1
#define IPP_VERSION_UPDATE 0
#endif

#define IPP_VERSION_X100 (IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR*10 + IPP_VERSION_UPDATE)

#ifdef HAVE_IPP_ICV
# define ICV_BASE
#if IPP_VERSION_X100 >= 201700
# include "ippicv.h"
#else
# include "ipp.h"
#endif
#else
# include "ipp.h"
#endif

#define CV_INSTRUMENT_FUN_IPP(FUN, ...) ((FUN)(__VA_ARGS__))

#endif

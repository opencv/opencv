// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "precomp.hpp"
#include "api/gmeta_check.hpp"

namespace cv
{
bool validate_input_meta_arg(const GMetaArg& meta, std::ostream* tracer)
{
    switch (meta.index())
    {
        case GMetaArg::index_of<GMatDesc>():
        {
            return validate_input_meta(util::get<GMatDesc>(meta), tracer);
        }
        default:
            break;
    }
    return true;
}

bool validate_input_meta(const GMatDesc& meta, std::ostream* tracer)
{
    if (meta.dims.empty())
    {
        if (!(meta.size.height > 0 && meta.size.width > 0))
        {
            if (tracer)
            {
                *tracer << "Image format is invalid. Size must contain positive values, got width: "
                        << meta.size.width << ", height: " << meta.size.height;
            }
            return false;
        }

        if (!(meta.chan > 0))
        {
            if (tracer)
            {
                *tracer << "Image format is invalid. Channel mustn't be negative value, got channel: "
                        << meta.chan;
            }
            return false;
        }
    }

    if (!(meta.depth >= 0))
    {
        if (tracer)
        {
            *tracer << "Image format is invalid. Depth must be positive value, got depth: "
                    << meta.depth;
        }
        return false;
    }
    return true;
}
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/util/throw.hpp>
#include "api/gmeta_check.hpp"

namespace cv
{
void validate_input_meta_arg(const GMetaArg& meta)
{
    switch (meta.index())
    {
        case GMetaArg::index_of<GMatDesc>():
        {
            validate_input_meta(util::get<GMatDesc>(meta));
        }
        default:
            break;
    }
}

void validate_input_meta(const GMatDesc& meta)
{
    if (meta.dims.empty())
    {
        if (!(meta.size.height > 0 && meta.size.width > 0))
        {
            util::throw_error(std::logic_error(std::string("Image format is invalid. Size must contain positive values, got width: ") +
                                               std::to_string(meta.size.width ) + (", height: ") +
                                               std::to_string(meta.size.height)));
        }

        if (!(meta.chan > 0))
        {
            util::throw_error(std::logic_error(std::string("Image format is invalid. Channel mustn't be negative value, got channel: ") +
                                               std::to_string(meta.chan)));
        }
    }

    if (!(meta.depth >= 0))
    {
        util::throw_error(std::logic_error(std::string("Image format is invalid. Depth must be positive value, got depth: ") +
                                           std::to_string(meta.depth)));
    }
    // All checks are ok
}
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <ade/util/zip_range.hpp>   // util::indexed
#include <opencv2/gapi/gcompoundkernel.hpp>
#include "compiler/gobjref.hpp"

// FIXME move to backends

cv::detail::GCompoundContext::GCompoundContext(const cv::GArgs& in_args)
{
    m_args.resize(in_args.size());
    for (const auto it : ade::util::indexed(in_args))
    {
        const auto& i      = ade::util::index(it);
        const auto& in_arg = ade::util::value(it);

        if (in_arg.kind != cv::detail::ArgKind::GOBJREF)
        {
            m_args[i] = in_arg;
        }
        else
        {
            const cv::gimpl::RcDesc &ref = in_arg.get<cv::gimpl::RcDesc>();
            switch (ref.shape)
            {
                case GShape::GMAT   : m_args[i] = GArg(GMat());    break;
                case GShape::GSCALAR: m_args[i] = GArg(GScalar()); break;
                case GShape::GARRAY :
                case GShape::GOPAQUE:
                    // do nothing - as handled in a special way, see gcompoundkernel.hpp for details
                    // same applies to GMatP
                    break;
                default: GAPI_Error("InternalError");
            }
        }
    }
    GAPI_Assert(m_args.size() == in_args.size());
}

cv::detail::GCompoundKernel::GCompoundKernel(const F& f) : m_f(f)
{
}

void cv::detail::GCompoundKernel::apply(cv::detail::GCompoundContext& ctx) { m_f(ctx); }

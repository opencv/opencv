// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include <type_traits>

#include "streaming/onevpl/engine/preproc/utils.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
namespace utils {

cv::MediaFormat fourcc_to_MediaFormat(int value) {
    switch (value)
    {
        case MFX_FOURCC_BGRP:
            return cv::MediaFormat::BGR;
        case MFX_FOURCC_NV12:
            return cv::MediaFormat::NV12;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported FourCC format requested: " << value <<
                                     ". Cannot cast to cv::MediaFrame");
            GAPI_Error("Unsupported FOURCC");

    }
}

int MediaFormat_to_fourcc(cv::MediaFormat value) {
    switch (value)
    {
        case cv::MediaFormat::BGR:
            return MFX_FOURCC_BGRP;
        case cv::MediaFormat::NV12:
            return MFX_FOURCC_NV12;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported cv::MediaFormat format requested: " <<
                                      static_cast<typename std::underlying_type<cv::MediaFormat>::type>(value) <<
                                     ". Cannot cast to FourCC");
            GAPI_Error("Unsupported cv::MediaFormat");
    }
}
int MediaFormat_to_chroma(cv::MediaFormat value) {
    switch (value)
    {
        case cv::MediaFormat::BGR:
            return MFX_CHROMAFORMAT_MONOCHROME;
        case cv::MediaFormat::NV12:
            return MFX_CHROMAFORMAT_YUV420;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported cv::MediaFormat format requested: " <<
                                      static_cast<typename std::underlying_type<cv::MediaFormat>::type>(value) <<
                                     ". Cannot cast to ChromaFormateIdc");
            GAPI_Error("Unsupported cv::MediaFormat");
    }
}

mfxFrameInfo to_mfxFrameInfo(const cv::GFrameDesc& frame_info) {
    mfxFrameInfo ret {0};
    ret.FourCC        = MediaFormat_to_fourcc(frame_info.fmt);
    ret.ChromaFormat  = MediaFormat_to_chroma(frame_info.fmt);
    ret.Width         = frame_info.size.width;
    ret.Height        = frame_info.size.height;
    ret.CropX         = 0;
    ret.CropY         = 0;
    ret.CropW         = 0;
    ret.CropH         = 0;
    ret.PicStruct     = MFX_PICSTRUCT_UNKNOWN;
    ret.FrameRateExtN = 0;
    ret.FrameRateExtD = 0;
    return ret;
}
} // namespace utils
} // namespace cv
} // namespace gapi
} // namespace wip
} // namespace onevpl

#endif // HAVE_ONEVPL

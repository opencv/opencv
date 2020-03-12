// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Stefan Brüns <stefan.bruens@rwth-aachen.de>

#include "precomp.hpp"

#ifdef HAVE_OPENJPEG
#include "grfmt_jpeg2000_openjpeg.hpp"

#include <sstream>

#include <opencv2/core/utils/logger.hpp>

#include "opencv2/imgproc.hpp"

namespace cv {

namespace {

const String colorspaceName(COLOR_SPACE colorspace) {
    switch(colorspace) {
        case OPJ_CLRSPC_CMYK:
            return "CMYK";
        case OPJ_CLRSPC_SRGB:
            return "sRGB";
        case OPJ_CLRSPC_EYCC:
            return "e-YCC";
        case OPJ_CLRSPC_GRAY:
            return "grayscale";
        case OPJ_CLRSPC_SYCC:
            return "YUV";
        case OPJ_CLRSPC_UNKNOWN:
            return "unknown";
        case OPJ_CLRSPC_UNSPECIFIED:
            return "unspecified";
        default:
            CV_Assert(!"Invalid colorspace");
    }
}


template<typename OutT, bool doShift, typename InT, std::size_t N>
void copy_data_(Mat& out, const std::array<InT*, N>& data, uint8_t shift) {
    Size size = out.size();
    if (out.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }

    std::array<const InT*, N> sptr = {nullptr, };
    for (int i = 0; i < size.height; i++ ) {
        for (std::size_t c = 0; c < N; c++) {
            sptr[c] = data[c] + i * size.width;
        }
        OutT* dptr = out.ptr<OutT>(i);
        // when the arrays are continuous,
        // the outer loop is executed only once
        for( int j = 0; j < size.width; j++) {
            for (std::size_t c = 0; c < N; c++) {
                InT in = *(sptr[c])++;
                if (doShift) {
                    *dptr++ = in >> shift;
                } else {
                    *dptr++ = in;
                }
            }
        }
    }
}

template<typename InT, std::size_t N>
void copy_data(Mat& out, const std::array<InT*, N>& data, uint8_t shift) {
    if (out.depth() == CV_8U && shift == 0)
        copy_data_<uint8_t, false>(out, data, 0);
    else if (out.depth() == CV_8U)
        copy_data_<uint8_t, true>(out, data, shift);
    else if (out.depth() == CV_16U)
        copy_data_<uint16_t, false>(out, data, 0);
    else
        CV_Error(Error::StsNotImplemented, "only depth CV_8U and CV16_U are supported");
}

} // namespace <anonymous>

/////////////////////// Jpeg2KOpjDecoder ///////////////////

Jpeg2KOpjDecoder::Jpeg2KOpjDecoder()
{
    static const unsigned char signature[] = { 0, 0, 0, 0x0c, 'j', 'P', ' ', ' ', 13, 10, 0x87, 10};
    m_signature = String((const char*)(signature), sizeof(signature));
    // FIXME: raw code stream
}

Jpeg2KOpjDecoder::~Jpeg2KOpjDecoder()
{
    opj_set_error_handler(codec_.get(), nullptr, nullptr);
}

ImageDecoder Jpeg2KOpjDecoder::newDecoder() const
{
    return makePtr<Jpeg2KOpjDecoder>();
}

void Jpeg2KOpjDecoder::setMessageHandlers()
{
    opj_msg_callback error_handler = [](const char* msg, void* client_data) {
        static_cast<Jpeg2KOpjDecoder*>(client_data)->m_errorMessage += msg;
    };

    opj_set_error_handler(codec_.get(), error_handler, this);
}

bool Jpeg2KOpjDecoder::readHeader()
{
    opj_dparameters parameters;
    opj_set_default_decoder_parameters(&parameters);

    stream_.reset(opj_stream_create_default_file_stream(m_filename.c_str(), OPJ_STREAM_READ));
    if (!stream_)
        return false;

    codec_.reset(opj_create_decompress(OPJ_CODEC_JP2));
    if (!codec_)
        return false;

    setMessageHandlers();

    if (!opj_setup_decoder(codec_.get(), &parameters))
        return false;

    {
        opj_image_t* rawImage;
        if (!opj_read_header(stream_.get(), codec_.get(), &rawImage))
            return false;

        image_.reset(rawImage);
    }

    m_width = image_->x1 - image_->x0;
    m_height = image_->y1 - image_->y0;

    /* Different components may have different precision,
     * so check all.
     */
    bool hasAlpha = false;
    const int numcomps = image_->numcomps;
    CV_Assert(numcomps >= 1);
    for (int i = 0; i < numcomps; i++) {
        const opj_image_comp_t& comp = image_->comps[i];
        
        if (comp.sgnd)
        {
            CV_Error(Error::StsNotImplemented, cv::format("component %d/%d is signed", i, numcomps));
        }
         

        if (hasAlpha && comp.alpha)
        {
            CV_Error(Error::StsNotImplemented, cv::format("componenet %d/%d is duplicate alpha channel", i, numcomps));
        }
            
        hasAlpha |= comp.alpha;

        m_maxPrec = std::max(m_maxPrec, comp.prec);
    }

    if (m_maxPrec < 8) {
        CV_Error(Error::StsNotImplemented, "precision < 8 not supported");
    } else if (m_maxPrec == 8) {
        m_type = CV_MAKETYPE(CV_8U, numcomps);
    } else if (m_maxPrec <= 16) {
        m_type = CV_MAKETYPE(CV_16U, numcomps);
    } else if (m_maxPrec <= 23) {
        m_type = CV_MAKETYPE(CV_32F, numcomps);
    } else {
        m_type = CV_MAKETYPE(CV_64F, numcomps);
    }
    return true;
}

bool Jpeg2KOpjDecoder::readData( Mat& img )
{
    m_errorMessage.clear();
    if (!opj_decode(codec_.get(), stream_.get(), image_.get())) {
        CV_Error(Error::StsError, "failed to decode:" + m_errorMessage);
    }

    if (image_->color_space == OPJ_CLRSPC_UNSPECIFIED)
        CV_Error(Error::StsNotImplemented, "image has unspecified color space");

    // file format
    const int numcomps = image_->numcomps;

    // requested format
    const int channels = CV_MAT_CN(img.type());
    const int depth = CV_MAT_DEPTH(img.type());
    const OPJ_UINT32 outPrec = [depth]() {
        if (depth == CV_8U) return 8;
        if (depth == CV_16U) return 16;
        CV_Error(Error::StsNotImplemented, "output precision > 16 not supported");
    }();
    const uint8_t shift = outPrec > m_maxPrec ? 0 : m_maxPrec - outPrec;

    if (image_->color_space == OPJ_CLRSPC_SRGB || image_->color_space == OPJ_CLRSPC_UNKNOWN) {
        // Assume gray (+ alpha) for 1 channels -> gray
        if (channels == 1 && numcomps <= 2) {
            const std::array<OPJ_INT32*, 1> incomps = {image_->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGB (+ alpha) for 3 channels -> BGR
        if (channels == 3 && (numcomps == 3 || numcomps == 4)) {
            const std::array<OPJ_INT32*, 3> incomps = {image_->comps[2].data, image_->comps[1].data, image_->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGBA for 4 channels -> BGRA
        if (channels == 4 && numcomps == 4) {
            const std::array<OPJ_INT32*, 4> incomps = {image_->comps[2].data, image_->comps[1].data, image_->comps[0].data, image_->comps[3].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGB for >= 3 channels -> gray
        if (channels == 1 && numcomps >= 3) {
            Mat tmp(img.size(), CV_MAKETYPE(depth, 3));
            const std::array<OPJ_INT32*, 3> incomps = {image_->comps[2].data, image_->comps[1].data, image_->comps[0].data};
            copy_data(tmp, incomps, shift);
            cvtColor(tmp, img, COLOR_BGR2GRAY);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
                 cv::format("Unsupported number of channels during color conversion, IN: %d OUT: %d",
                            numcomps, channels));

    } else if (image_->color_space == OPJ_CLRSPC_GRAY) {
        if (channels == 3) {
            const std::array<OPJ_INT32*, 3> incomps = {image_->comps[0].data, image_->comps[0].data, image_->comps[0].data};
            copy_data(img, incomps, shift);
            return true;

        } else if (channels == 1) {
            const std::array<OPJ_INT32*, 1> incomps = {image_->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
                 cv::format("Unsupported number of channels during color conversion, IN: %d OUT %d",
                            numcomps, channels));

    } else if (image_->color_space == OPJ_CLRSPC_SYCC) {
        if (channels == 1) {
            const std::array<OPJ_INT32*, 1> incomps = {image_->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        if (channels == 3 && numcomps >= 3) {
            const std::array<OPJ_INT32*, 3> incomps = {image_->comps[0].data, image_->comps[1].data, image_->comps[2].data};
            copy_data(img, incomps, shift);
            cvtColor(img, img, COLOR_YUV2BGR);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
                 cv::format("Unsupported number of channels during color conversion, IN: %d OUT %d",
                            numcomps, channels));
    }

    CV_Error(Error::StsNotImplemented, cv::format("Unsupported color space conversion: %s -> %s",
                                                  colorspaceName(image_->color_space).c_str(),(channels == 1) ? "gray" : "BGR"));

    return false;
}

} // namespace cv

#endif

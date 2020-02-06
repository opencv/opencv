// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Stefan Brüns <stefan.bruens@rwth-aachen.de>

#include "precomp.hpp"

#ifdef HAVE_OPENJPEG
#include <sstream>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "grfmt_jpeg2000_openjpeg.hpp"
#include "opencv2/imgproc.hpp"

#include <openjpeg.h>

namespace cv {

namespace {
struct streamDeleter {
    void operator()(opj_stream_t* obj) const {
        if (obj) opj_stream_destroy(obj);
    }
};

struct codecDeleter {
    void operator()(opj_codec_t* obj) const {
        if (obj) opj_destroy_codec(obj);
    }
};

struct imageDeleter {
    void operator()(opj_image_t* obj) const {
        if (obj) opj_image_destroy(obj);
    }
};

const String colorspaceName(int colorspace) {
    if (colorspace == OPJ_CLRSPC_UNKNOWN)
        return "unknown";
    if (colorspace == OPJ_CLRSPC_UNSPECIFIED)
        return "unspecified";
    if (colorspace == OPJ_CLRSPC_SRGB)
        return "sRGB";
    if (colorspace == OPJ_CLRSPC_GRAY)
        return "grayscale";
    if (colorspace == OPJ_CLRSPC_SYCC)
        return "YUV";
    if (colorspace == OPJ_CLRSPC_EYCC)
        return "e-YCC";
    if (colorspace == OPJ_CLRSPC_CMYK)
        return "CMYK";
    return "bad colorspace value";
}


template<typename OutT, bool doShift, typename InT, int N>
void copy_data_(Mat& out, const InT* (&data)[N], uint8_t shift) {
    Size size = out.size();
    if (out.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }

    const InT* sptr[N];
    for (int i = 0; i < size.height; i++ ) {
        for (int c = 0; c < N; c++) {
            sptr[c] = static_cast<const InT*>(data[c]) + i * size.width;
        }
        OutT* dptr = out.ptr<OutT>(i);
        // when the arrays are continuous,
        // the outer loop is executed only once
        for( int j = 0; j < size.width; j++) {
            for (int c = 0; c < N; c++) {
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

template<typename InT, int N>
void copy_data(Mat& out, const InT* (&data)[N], uint8_t shift) {
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
    opj_set_error_handler(m_codec, nullptr, nullptr);
}

size_t Jpeg2KOpjDecoder::signatureLength() const
{
    return m_signature.size();
}

bool Jpeg2KOpjDecoder::checkSignature( const String& signature ) const
{
    if (signature.size() >= m_signature.size() &&
        0 == signature.compare(0, m_signature.size(), m_signature))
    {
        return true;
    }

    return false;
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

    opj_set_error_handler(m_codec, error_handler, this);
}

bool Jpeg2KOpjDecoder::readHeader()
{
    opj_dparameters parameters;
    opj_set_default_decoder_parameters(&parameters);

    std::unique_ptr<opj_stream_t, streamDeleter> stream;
    std::unique_ptr<opj_codec_t, codecDeleter> codec;
    std::unique_ptr<opj_image_t, imageDeleter> image;

    stream.reset(opj_stream_create_default_file_stream(m_filename.c_str(), OPJ_STREAM_READ));
    if (!stream)
        return false;

    codec.reset(opj_create_decompress(OPJ_CODEC_JP2));
    if (!codec)
        return false;

    setMessageHandlers();

    if (!opj_setup_decoder(codec.get(), &parameters))
        return false;

    {
        opj_image_t* _image;
        if (!opj_read_header(stream.get(), codec.get(), &_image))
            return false;

        image.reset(_image);
    }

    m_width = image->x1 - image->x0;
    m_height = image->y1 - image->y0;

    /* Different components may have different precision,
     * so check all.
     */
    bool hasAlpha = false;
    int numcomps = image->numcomps;
    CV_Assert(numcomps >= 1);
    for (int i = 0; i < numcomps; i++) {
        const opj_image_comp_t& comp = image->comps[i];
        auto msgprefix = [&]() { return "component " + std::to_string(i) + "/" + std::to_string(numcomps); };
        if (comp.sgnd)
            CV_Error(Error::StsNotImplemented, msgprefix() + " is signed");

        if (hasAlpha && comp.alpha)
            CV_Error(Error::StsNotImplemented, msgprefix() + " is duplicate alpha channel");
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

    m_image = image.release();
    m_codec = codec.release();
    m_stream = stream.release();

    return true;
}

bool Jpeg2KOpjDecoder::readData( Mat& img )
{
    std::unique_ptr<opj_stream_t, streamDeleter> stream{m_stream};
    std::unique_ptr<opj_codec_t, codecDeleter> codec{m_codec};
    std::unique_ptr<opj_image_t, imageDeleter> image{m_image};

    m_errorMessage.clear();
    if (!opj_decode(m_codec, m_stream, m_image)) {
        CV_Error(Error::StsError, "failed to decode:" + m_errorMessage);
    }

    if (m_image->color_space == OPJ_CLRSPC_UNSPECIFIED)
        CV_Error(Error::StsNotImplemented, "image has unspecified color space");

    // file format
    const int numcomps = image->numcomps;

    // requested format
    const int channels = CV_MAT_CN(img.type());
    const int depth = CV_MAT_DEPTH(img.type());
    const OPJ_UINT32 outPrec = [depth]() {
        if (depth == CV_8U) return 8;
        if (depth == CV_16U) return 16;
        CV_Error(Error::StsNotImplemented, "output precision > 16 not supported");
    }();
    const uint8_t shift = outPrec > m_maxPrec ? 0 : m_maxPrec - outPrec;

    if (m_image->color_space == OPJ_CLRSPC_SRGB || m_image->color_space == OPJ_CLRSPC_UNKNOWN) {
        // Assume gray (+ alpha) for 1 channels -> gray
        if (channels == 1 && numcomps <= 2) {
            const OPJ_INT32* incomps[] = {image->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGB (+ alpha) for 3 channels -> BGR
        if (channels == 3 && (numcomps == 3 || numcomps == 4)) {
            const OPJ_INT32* incomps[] = {image->comps[2].data, image->comps[1].data, image->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGBA for 4 channels -> BGRA
        if (channels == 4 && numcomps == 4) {
            const OPJ_INT32* incomps[] = {image->comps[2].data, image->comps[1].data, image->comps[0].data, image->comps[3].data};
            copy_data(img, incomps, shift);
            return true;
        }

        // Assume RGB for >= 3 channels -> gray
        if (channels == 1 && numcomps >= 3) {
            Mat tmp(img.size(), CV_MAKETYPE(depth, 3));
            const OPJ_INT32* incomps[] = {image->comps[2].data, image->comps[1].data, image->comps[0].data};
            copy_data(tmp, incomps, shift);
            cvtColor(tmp, img, COLOR_BGR2GRAY);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
            "unsupported number of channels during color conversion, IN: "
            + std::to_string(numcomps) + " OUT: " + std::to_string(channels));

    } else if (m_image->color_space == OPJ_CLRSPC_GRAY) {
        if (channels == 3) {
            const OPJ_INT32* incomps[] = {image->comps[0].data, image->comps[0].data, image->comps[0].data};
            copy_data(img, incomps, shift);
            return true;

        } else if (channels == 1) {
            const OPJ_INT32* incomps[] = {image->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
            "unsupported number of channels during color conversion, IN: "
            + std::to_string(numcomps) + " OUT: " + std::to_string(channels));

    } else if (m_image->color_space == OPJ_CLRSPC_SYCC) {
        if (channels == 1) {
            const OPJ_INT32* incomps[] = {image->comps[0].data};
            copy_data(img, incomps, shift);
            return true;
        }

        if (channels == 3 && numcomps >= 3) {
            const OPJ_INT32* incomps[] = {image->comps[0].data, image->comps[1].data, image->comps[2].data};
            copy_data(img, incomps, shift);
            cvtColor(img, img, COLOR_YUV2BGR);
            return true;
        }

        CV_Error(Error::StsNotImplemented,
            "unsupported number of channels during color conversion, IN: "
            + std::to_string(numcomps) + " OUT: " + std::to_string(channels));
    }

    CV_Error(Error::StsNotImplemented, "unsupported color space conversion: "
        + colorspaceName(m_image->color_space) + " -> "
        + (channels == 1 ? "gray" : "BGR"));

    return false;
}

} // namespace cv

#endif

/* End of file. */

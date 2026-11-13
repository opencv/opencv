// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "grfmt_htj2k.hpp"

#ifdef HAVE_OPENHTJ2K

#include <opencv2/core/utils/logger.hpp>

#include <cstdio>
#include <exception>
#include <functional>

// OpenHTJ2K public API (include directory provided by the openhtj2k::openhtj2k target)
#include <open_htj2k/interface/decoder.hpp>
#include <open_htj2k/interface/encoder.hpp>
#include <open_htj2k/interface/openhtj2k_message.hpp>

namespace cv
{

namespace
{

// Enough bytes to reach the CAP marker of a raw codestream with up to 150 components
// (the CAP marker follows the SIZ marker segment, whose length grows by 3 bytes per component).
const size_t kHTJ2KSignatureLength = 512;

// ISO/IEC 15444-1 Annex I: the JPEG 2000 signature box that opens every JP2-family file.
const uchar kJP2SignatureBox[12] = { 0x00, 0x00, 0x00, 0x0C, 'j', 'P', ' ', ' ', 0x0D, 0x0A, 0x87, 0x0A };

// Codestream markers (ISO/IEC 15444-1 Annex A).
const int kMarkerSIZ = 0x51;
const int kMarkerCAP = 0x50;
const int kMarkerSOT = 0x90;
const int kMarkerEOC = 0xD9;

// Rsiz bit 14: a CAP marker segment is present.  Pcap bit for Part 15 (HTJ2K).
const unsigned kRsizCapPresent = 0x4000;
const unsigned kPcapPart15 = 1u << (32 - 15);

// "no Qfactor" value of the OpenHTJ2K encoder API.
const uint8_t kNoQfactor = 0xFF;

inline unsigned readBE16(const uchar* p) { return ((unsigned)p[0] << 8) | p[1]; }
inline unsigned readBE32(const uchar* p) { return (readBE16(p) << 16) | readBE16(p + 2); }

// JPH file (ISO/IEC 15444-15 Annex D): the file type box names "jph " as the brand or
// in its compatibility list.
bool isJPHFile(const uchar* p, size_t n)
{
    if (n < 28 || memcmp(p, kJP2SignatureBox, sizeof(kJP2SignatureBox)) != 0)
        return false;
    if (memcmp(p + 16, "ftyp", 4) != 0)
        return false;
    const size_t box_end = std::min(n, (size_t)12 + readBE32(p + 12));
    if (memcmp(p + 20, "jph ", 4) == 0)
        return true;
    for (size_t off = 28; off + 4 <= box_end; off += 4)  // compatibility list
    {
        if (memcmp(p + off, "jph ", 4) == 0)
            return true;
    }
    return false;
}

// Raw codestream that uses the HT block coder: Rsiz announces a CAP marker segment and
// the CAP marker signals Part 15.
bool isHTCodestream(const uchar* p, size_t n)
{
    if (n < 8 || p[0] != 0xFF || p[1] != 0x4F || p[2] != 0xFF || p[3] != kMarkerSIZ)
        return false;
    if ((readBE16(p + 6) & kRsizCapPresent) == 0)
        return false;
    // Walk the main-header marker segments that follow SIZ, looking for CAP.
    size_t off = 4 + readBE16(p + 4);
    while (off + 4 <= n)
    {
        if (p[off] != 0xFF || p[off + 1] == kMarkerSOT)
            return false;
        const size_t len = readBE16(p + off + 2);
        if (p[off + 1] == kMarkerCAP)
            return off + 8 <= n && (readBE32(p + off + 4) & kPcapPart15) != 0;
        if (len < 2)
            return false;
        off += 2 + len;
    }
    return false;
}

// The contiguous codestream box (ISO/IEC 15444-1 I.5.4) of a JP2-family file, as [begin, end).
// The boxes before it must lie within the data.
bool findCodestreamBox(const uchar* p, size_t n, size_t& begin, size_t& end)
{
    size_t off = 0;
    while (off + 8 <= n)
    {
        uint64_t len = readBE32(p + off);
        size_t hdr = 8;
        if (len == 1)  // the length is in the XLBox field
        {
            if (off + 16 > n)
                return false;
            len = ((uint64_t)readBE32(p + off + 8) << 32) | readBE32(p + off + 12);
            hdr = 16;
        }
        else if (len == 0)  // the box runs to the end of the file
        {
            len = n - off;
        }
        if (len < hdr || len > n - off)
            return false;
        if (memcmp(p + off + 4, "jp2c", 4) == 0)
        {
            begin = off + hdr;
            end = off + (size_t)len;
            return true;
        }
        off += (size_t)len;
    }
    return false;
}

// True when the codestream holds every tile-part it announces and ends with the EOC marker.
// The decoder decodes a truncated codestream in part; this adapter refuses it, as the other
// decoders do, rather than return an image with missing data.
bool isCompleteCodestream(const uchar* p, size_t n)
{
    if (n < 4 || p[0] != 0xFF || p[1] != 0x4F)
        return false;
    size_t off = 2;
    while (off + 4 <= n && p[off] == 0xFF && p[off + 1] != kMarkerSOT)  // main header
    {
        const size_t len = readBE16(p + off + 2);
        if (len < 2)
            return false;
        off += 2 + len;
    }
    // Tile-parts. Psot is the length from the SOT marker to the end of the tile-part, or 0 for
    // a last tile-part that runs to the EOC marker (ISO/IEC 15444-1 A.4.2).
    while (off + 12 <= n && p[off] == 0xFF && p[off + 1] == kMarkerSOT)
    {
        const size_t psot = readBE32(p + off + 6);
        if (psot == 0)
            return n >= off + 14 + 2 && p[n - 2] == 0xFF && p[n - 1] == kMarkerEOC;
        if (psot < 14 || psot > n - off)  // SOT and SOD markers take 14 bytes
            return false;
        off += psot;
    }
    return off + 2 <= n && p[off] == 0xFF && p[off + 1] == kMarkerEOC;
}

uint32_t htj2kThreads()
{
    return (uint32_t)std::max(1, cv::getNumThreads());
}

// OpenHTJ2K reports errors as exceptions and sends warnings to a process-wide handler.
void htj2kMessageHandler(open_htj2k::message_level level, const char* text, void* /*user_data*/)
{
    if (level == open_htj2k::message_level::warning)
    {
        CV_LOG_WARNING(NULL, "HTJ2K: " << text);
    }
    else
    {
        CV_LOG_INFO(NULL, "HTJ2K: " << text);
    }
}

bool htj2kSetMessageHandler_()
{
    open_htj2k::set_message_handler(htj2kMessageHandler, nullptr);
    return true;
}

void htj2kSetMessageHandler()
{
    static bool v = htj2kSetMessageHandler_();
    CV_UNUSED(v);
}

bool readWholeFile(const String& filename, std::vector<uchar>& data)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f)
        return false;
    uchar chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0)
        data.insert(data.end(), chunk, chunk + n);
    const bool ok = ferror(f) == 0;
    fclose(f);
    return ok;
}

// How decoded components map to the channels of the output image.
struct RowLayout
{
    int out_cn;
    int src[4];     // component index for each output channel
    bool to_gray;   // output = luma of components 0, 1, 2 (R, G, B)
    int add;        // added before the shift (signed input to unsigned output)
    int shift;      // right shift from the source precision to the output depth
};

template <typename T>
void storeRow(T* dst, int width, const int32_t* const* comps, const RowLayout& L)
{
    const int add = L.add, shift = L.shift;
    if (L.to_gray)
    {
        // Same fixed-point weights as cvtColor(COLOR_RGB2GRAY).
        const int32_t *r = comps[0], *g = comps[1], *b = comps[2];
        for (int x = 0; x < width; x++)
        {
            const int R = (r[x] + add) >> shift, G = (g[x] + add) >> shift, B = (b[x] + add) >> shift;
            dst[x] = saturate_cast<T>((R * 4899 + G * 9617 + B * 1868 + (1 << 13)) >> 14);
        }
        return;
    }
    const int cn = L.out_cn;
    for (int k = 0; k < cn; k++)
    {
        const int32_t* s = comps[L.src[k]];
        T* d = dst + k;
        for (int x = 0; x < width; x++, d += cn)
            *d = saturate_cast<T>((s[x] + add) >> shift);
    }
}

// Channel k of an image row, as samples of one component.
template <typename T>
void loadRow(int32_t* dst, const Mat& img, int y, int k)
{
    const int cn = img.channels();
    const T* s = img.ptr<T>(y) + k;
    for (int x = 0; x < img.cols; x++, s += cn)
        dst[x] = *s;
}

// sYCC (ISO/IEC 15444-1 Annex I, EnumCS 18) to RGB, in the source precision.
void yccToRgb(const int32_t* const* ycc, int width, int precision, int32_t* r, int32_t* g, int32_t* b)
{
    const float center = (float)(1 << (precision - 1));
    const int maxv = (1 << precision) - 1;
    for (int x = 0; x < width; x++)
    {
        const float Y = (float)ycc[0][x], Cb = ycc[1][x] - center, Cr = ycc[2][x] - center;
        r[x] = std::min(std::max(cvRound(Y + 1.402f * Cr), 0), maxv);
        g[x] = std::min(std::max(cvRound(Y - 0.344136f * Cb - 0.714136f * Cr), 0), maxv);
        b[x] = std::min(std::max(cvRound(Y + 1.772f * Cb), 0), maxv);
    }
}

}  // namespace

/////////////////////// HTJ2KDecoder ///////////////////

HTJ2KDecoder::HTJ2KDecoder()
    : m_num_components(0), m_precision(0), m_signed(false), m_ycc(false)
{
    m_buf_supported = true;
}

HTJ2KDecoder::~HTJ2KDecoder()
{
}

size_t HTJ2KDecoder::signatureLength() const
{
    return kHTJ2KSignatureLength;
}

bool HTJ2KDecoder::checkSignature(const String& signature) const
{
    const uchar* p = (const uchar*)signature.c_str();
    const size_t n = signature.size();
    return isJPHFile(p, n) || isHTCodestream(p, n);
}

ImageDecoder HTJ2KDecoder::newDecoder() const
{
    htj2kSetMessageHandler();
    return makePtr<HTJ2KDecoder>();
}

bool HTJ2KDecoder::readHeader()
{
    std::vector<uchar> file_data;
    const uchar* data;
    size_t size;
    if (!m_buf.empty())
    {
        CV_Assert(m_buf.isContinuous());
        data = m_buf.ptr();
        size = m_buf.total() * m_buf.elemSize();
    }
    else
    {
        if (!readWholeFile(m_filename, file_data))
            return false;
        data = file_data.data();
        size = file_data.size();
    }

    size_t cs_begin = 0, cs_end = size;
    if (isJPHFile(data, size) && !findCodestreamBox(data, size, cs_begin, cs_end))
    {
        CV_LOG_WARNING(NULL, "HTJ2K: the file is truncated or has no codestream box");
        return false;
    }
    if (!isCompleteCodestream(data + cs_begin, cs_end - cs_begin))
    {
        CV_LOG_WARNING(NULL, "HTJ2K: the codestream is truncated");
        return false;
    }

    try
    {
        m_decoder.reset(new open_htj2k::openhtj2k_decoder());
        m_decoder->init(data, size, 0, htj2kThreads());  // init() copies the data
        m_decoder->parse();
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "HTJ2K: cannot parse the codestream: " << e.what());
        m_decoder.reset();
        return false;
    }

    const int nc = m_decoder->get_num_component();
    if (nc < 1 || nc > 4)
        CV_Error(Error::StsNotImplemented, cv::format("HTJ2K: %d components are not supported (1 to 4)", nc));
    const uint32_t width = m_decoder->get_component_width(0);
    const uint32_t height = m_decoder->get_component_height(0);
    const bool is_signed = m_decoder->get_component_signedness(0);
    int precision = 0;
    for (int c = 0; c < nc; c++)
    {
        if (m_decoder->get_component_width((uint16_t)c) != width ||
            m_decoder->get_component_height((uint16_t)c) != height)
            CV_Error(Error::StsNotImplemented, "HTJ2K: sub-sampled components are not supported");
        if (m_decoder->get_component_signedness((uint16_t)c) != is_signed)
            CV_Error(Error::StsNotImplemented, "HTJ2K: components of mixed signedness are not supported");
        precision = std::max(precision, (int)m_decoder->get_component_depth((uint16_t)c));
    }
    if (precision < 8 || precision > 16)
        CV_Error(Error::StsNotImplemented,
                 cv::format("HTJ2K: %d-bit components are not supported (8 to 16 bits)", precision));
    if (width == 0 || height == 0 || width > (uint32_t)INT_MAX || height > (uint32_t)INT_MAX)
        return false;

    m_num_components = nc;
    m_precision = precision;
    m_signed = is_signed;
    // EnumCS of the JPH colour specification box; 0 for a raw codestream, or for a box that is not
    // an enumerated colour space (an ICC profile), whose components are taken as gray (+ alpha) or
    // RGB (+ alpha) by their count, as the OpenJPEG backend does for an unspecified colour space.
    // Other colour spaces are not converted here, so they are refused rather than returned as RGB.
    const uint32_t cs = m_decoder->get_colorspace();
    const bool cs_ok = cs == 0 || (cs == open_htj2k::ENUMCS_GRAYSCALE && nc <= 2) ||
                       ((cs == open_htj2k::ENUMCS_SRGB || cs == open_htj2k::ENUMCS_YCBCR) && nc >= 3);
    if (!cs_ok)
        CV_Error(Error::StsNotImplemented,
                 cv::format("HTJ2K: colour space %u with %d components is not supported", (unsigned)cs, nc));
    m_ycc = cs == open_htj2k::ENUMCS_YCBCR;
    if (m_ycc && m_signed)
        CV_Error(Error::StsNotImplemented, "HTJ2K: signed YCbCr components are not supported");

    m_width = (int)width;
    m_height = (int)height;
    const int depth = m_signed ? (precision <= 8 ? CV_8S : CV_16S) : (precision <= 8 ? CV_8U : CV_16U);
    // 2 components are gray + alpha, returned as BGRA like PNG does.
    m_type = CV_MAKETYPE(depth, nc == 1 ? 1 : nc == 2 ? 4 : nc);
    return true;
}

bool HTJ2KDecoder::readData(Mat& img)
{
    if (!m_decoder)
        return false;
    CV_Assert(img.cols == m_width && img.rows == m_height);

    const int nc = m_num_components;
    const int out_cn = img.channels();
    const int out_depth = img.depth();

    RowLayout L;
    L.out_cn = out_cn;
    L.to_gray = out_cn == 1 && nc >= 3 && !m_ycc;
    if (out_cn == 1)
    {
        L.src[0] = 0;  // gray, or Y of YCbCr
    }
    else if (out_cn == 3 || (out_cn == 4 && (nc == 2 || nc == 4)))
    {
        if (nc <= 2)
        {
            L.src[0] = L.src[1] = L.src[2] = 0;
            L.src[3] = 1;
        }
        else
        {
            L.src[0] = m_use_rgb ? 0 : 2;
            L.src[1] = 1;
            L.src[2] = m_use_rgb ? 2 : 0;
            L.src[3] = 3;
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented,
                 cv::format("HTJ2K: cannot output %d channels from %d components", out_cn, nc));
    }

    int out_bits;
    switch (out_depth)
    {
    case CV_8U: case CV_8S: out_bits = 8; break;
    case CV_16U: case CV_16S: out_bits = 16; break;
    default:
        CV_Error(Error::StsNotImplemented, cv::format("HTJ2K: unsupported output depth %d", out_depth));
    }
    const bool out_signed = out_depth == CV_8S || out_depth == CV_16S;
    L.add = (m_signed && !out_signed) ? (1 << (m_precision - 1)) : 0;
    L.shift = std::max(0, m_precision - out_bits);

    // YCbCr is converted to RGB for colour output; gray output takes Y as it is.
    const bool ycc = m_ycc && out_cn > 1;
    std::vector<int32_t> rgb;
    if (ycc)
        rgb.resize((size_t)3 * m_width);

    const int width = m_width;
    const int height = m_height;
    const int precision = m_precision;
    auto on_row = [&](uint32_t y, int32_t* const* rows, uint16_t ncomp)
    {
        if (y >= (uint32_t)height || ncomp < nc)
            return;
        const int32_t* comps[4] = { rows[0], nc > 1 ? rows[1] : rows[0], nc > 2 ? rows[2] : rows[0],
                                    nc > 3 ? rows[3] : rows[0] };
        if (ycc)
        {
            int32_t* r = rgb.data();
            yccToRgb(comps, width, precision, r, r + width, r + 2 * width);
            comps[0] = r;
            comps[1] = r + width;
            comps[2] = r + 2 * width;
        }
        switch (out_depth)
        {
        case CV_8U: storeRow(img.ptr<uchar>((int)y), width, comps, L); break;
        case CV_8S: storeRow(img.ptr<schar>((int)y), width, comps, L); break;
        case CV_16U: storeRow(img.ptr<ushort>((int)y), width, comps, L); break;
        case CV_16S: storeRow(img.ptr<short>((int)y), width, comps, L); break;
        }
    };

    std::vector<uint32_t> w, h;
    std::vector<uint8_t> d;
    std::vector<bool> s;
    try
    {
        m_decoder->invoke_line_based_stream(on_row, w, h, d, s);
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "HTJ2K: decoding failed: " << e.what());
        m_decoder.reset();
        return false;
    }
    m_decoder.reset();
    return true;
}

/////////////////////// HTJ2KEncoder ///////////////////

HTJ2KEncoder::HTJ2KEncoder(bool jph) : m_jph(jph)
{
    m_description = jph ? "HTJ2K files (*.jph)" : "HTJ2K codestreams (*.jhc)";
    m_buf_supported = true;
    m_supported_encode_key = { IMWRITE_HTJ2K_QUALITY_SCALE };
}

HTJ2KEncoder::~HTJ2KEncoder()
{
}

ImageEncoder HTJ2KEncoder::newEncoder() const
{
    htj2kSetMessageHandler();
    return makePtr<HTJ2KEncoder>(m_jph);
}

bool HTJ2KEncoder::isFormatSupported(int depth) const
{
    return depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S;
}

bool HTJ2KEncoder::write(const Mat& img, const std::vector<int>& params)
{
    const int depth = img.depth();
    const int cn = img.channels();
    CV_CheckDepth(depth, depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S,
                  "HTJ2K encoder supports CV_8U, CV_8S, CV_16U and CV_16S");
    CV_CheckChannels(cn, cn == 1 || cn == 3 || cn == 4, "HTJ2K encoder supports 1, 3 and 4 channels");

    int quality = 100;
    for (size_t i = 0; i + 1 < params.size(); i += 2)
    {
        if (params[i] == IMWRITE_HTJ2K_QUALITY_SCALE)
        {
            quality = std::min(std::max(params[i + 1], 0), 100);
            if (quality != params[i + 1])
                CV_LOG_WARNING(NULL, cv::format("HTJ2K: IMWRITE_HTJ2K_QUALITY_SCALE must be between 0 and 100, "
                                                "%d is used", quality));
        }
    }
    const bool lossless = quality == 100;
    const int bits = (depth == CV_8U || depth == CV_8S) ? 8 : 16;
    const bool is_signed = depth == CV_8S || depth == CV_16S;

    open_htj2k::siz_params siz;
    siz.Rsiz = 0;
    siz.Xsiz = (uint32_t)img.cols;
    siz.Ysiz = (uint32_t)img.rows;
    siz.XOsiz = siz.YOsiz = 0;
    siz.XTsiz = (uint32_t)img.cols;  // one tile
    siz.YTsiz = (uint32_t)img.rows;
    siz.XTOsiz = siz.YTOsiz = 0;
    siz.Csiz = (uint16_t)cn;
    for (int c = 0; c < cn; c++)
    {
        siz.Ssiz.push_back((uint8_t)((bits - 1) | (is_signed ? 0x80 : 0)));
        siz.XRsiz.push_back(1);
        siz.YRsiz.push_back(1);
    }

    open_htj2k::cod_params cod;
    cod.blkwidth = 4;  // 64x64 code-blocks (log2 - 2)
    cod.blkheight = 4;
    cod.is_max_precincts = true;
    cod.use_SOP = false;
    cod.use_EPH = false;
    cod.progression_order = 0;  // LRCP
    cod.number_of_layers = 1;
    cod.use_color_trafo = cn >= 3 ? 1 : 0;  // RCT (lossless) or ICT (lossy) on R, G, B; not on alpha
    cod.dwt_levels = 5;
    cod.codeblock_style = 0x40;  // HT block coder
    cod.transformation = lossless ? 1 : 0;  // 1: reversible 5/3, 0: irreversible 9/7

    open_htj2k::qcd_params qcd;
    qcd.number_of_guardbits = 1;
    qcd.is_derived = false;
    qcd.base_step = 1.0 / (double)(1 << bits);  // unused when lossless or with a Qfactor

    // Components in R, G, B(, A) order.
    const int src_ch[4] = { cn >= 3 ? 2 : 0, 1, 0, 3 };

    std::vector<uint8_t> out;
    try
    {
        std::vector<int32_t*> no_input;  // samples are supplied row by row below
        open_htj2k::openhtj2k_encoder encoder("", no_input, siz, cod, qcd,
                                              lossless ? kNoQfactor : (uint8_t)quality,
                                              m_jph, 0 /* sRGB */, htj2kThreads());
        if (!lossless)
            encoder.set_qfactor_weights(1);  // no visual weighting: every subband weight is 1
        encoder.set_output_buffer(out);
        encoder.invoke_line_based_stream([&](uint32_t y, int32_t** rows, uint16_t nc)
        {
            for (int c = 0; c < nc; c++)
            {
                switch (depth)
                {
                case CV_8U: loadRow<uchar>(rows[c], img, (int)y, src_ch[c]); break;
                case CV_8S: loadRow<schar>(rows[c], img, (int)y, src_ch[c]); break;
                case CV_16U: loadRow<ushort>(rows[c], img, (int)y, src_ch[c]); break;
                case CV_16S: loadRow<short>(rows[c], img, (int)y, src_ch[c]); break;
                }
            }
        });
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "HTJ2K: encoding failed: " << e.what());
        return false;
    }

    if (m_buf)
    {
        m_buf->assign(out.begin(), out.end());
        return true;
    }
    FILE* f = fopen(m_filename.c_str(), "wb");
    if (!f)
        return false;
    const bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
    return (fclose(f) == 0) && ok;
}

}  // namespace cv

#endif  // HAVE_OPENHTJ2K

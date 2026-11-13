// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGCODECS_GRFMT_HTJ2K_HPP
#define OPENCV_IMGCODECS_GRFMT_HTJ2K_HPP

#ifdef HAVE_OPENHTJ2K

#include "grfmt_base.hpp"
#include <memory>

namespace open_htj2k {
class openhtj2k_decoder;
}

namespace cv
{

/**
 * @brief HTJ2K (High-Throughput JPEG 2000, ISO/IEC 15444-15 | ITU-T T.814) decoder using OpenHTJ2K.
 *
 * Only HT content is claimed: JPH files, and raw codestreams whose CAP marker signals Part 15.
 * Other JPEG 2000 files are left to the JPEG 2000 decoder (OpenJPEG or JasPer).
 */
class HTJ2KDecoder CV_FINAL : public BaseImageDecoder
{
public:
    HTJ2KDecoder();
    ~HTJ2KDecoder() CV_OVERRIDE;

    bool readHeader() CV_OVERRIDE;
    bool readData(Mat& img) CV_OVERRIDE;
    size_t signatureLength() const CV_OVERRIDE;
    bool checkSignature(const String& signature) const CV_OVERRIDE;
    ImageDecoder newDecoder() const CV_OVERRIDE;

private:
    std::unique_ptr<open_htj2k::openhtj2k_decoder> m_decoder;
    int m_num_components;
    int m_precision;
    bool m_signed;
    bool m_ycc;
};

/**
 * @brief HTJ2K encoder using OpenHTJ2K.
 *
 * Writes a JPH file (*.jph) or a raw HTJ2K codestream (*.jhc). Lossless (reversible 5/3 wavelet)
 * by default; IMWRITE_HTJ2K_QUALITY_SCALE below 100 selects lossy coding (irreversible 9/7 wavelet)
 * without visual weighting.
 */
class HTJ2KEncoder CV_FINAL : public BaseImageEncoder
{
public:
    explicit HTJ2KEncoder(bool jph = true);
    ~HTJ2KEncoder() CV_OVERRIDE;

    bool isFormatSupported(int depth) const CV_OVERRIDE;
    bool write(const Mat& img, const std::vector<int>& params) CV_OVERRIDE;
    ImageEncoder newEncoder() const CV_OVERRIDE;

private:
    bool m_jph;
};

}  // namespace cv

#endif  // HAVE_OPENHTJ2K

#endif  // OPENCV_IMGCODECS_GRFMT_HTJ2K_HPP

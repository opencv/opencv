// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef __OPENCV_BARCODE_EAN13_DECODER_HPP__
#define __OPENCV_BARCODE_EAN13_DECODER_HPP__

#include "upcean_decoder.hpp"

namespace cv {
namespace barcode {
//extern struct EncodePair;
using std::string;
using std::vector;
using std::pair;


class Ean13Decoder : public UPCEANDecoder
{
public:
    Ean13Decoder();

    ~Ean13Decoder() override = default;

protected:
    Result decode(const vector<uchar> &data) const override;
};
}
} // namespace cv
#endif // !__OPENCV_BARCODE_EAN13_DECODER_HPP__

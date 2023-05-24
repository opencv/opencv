// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef __OPENCV_BARCODE_EAN8_DECODER_HPP__
#define __OPENCV_BARCODE_EAN8_DECODER_HPP__

#include "upcean_decoder.hpp"

namespace cv {
namespace barcode {

using std::string;
using std::vector;
using std::pair;

class Ean8Decoder : public UPCEANDecoder
{

public:
    Ean8Decoder();

    ~Ean8Decoder() override = default;

protected:
    Result decode(const vector<uchar> &data) const override;
};
}
}

#endif //__OPENCV_BARCODE_EAN8_DECODER_HPP__

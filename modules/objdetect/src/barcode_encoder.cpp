// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"
#include <climits>
#include <limits>

namespace cv {
namespace barcode {

static const char* const PATTERNS_A[10] = {
    "0001101", // 0
    "0011001", // 1
    "0010011", // 2
    "0111101", // 3
    "0100011", // 4
    "0110001", // 5
    "0101111", // 6
    "0111011", // 7
    "0110111", // 8
    "0001011"  // 9
};

static const char* const PATTERNS_B[10] = {
    "0100111", // 0
    "0110011", // 1
    "0011011", // 2
    "0100001", // 3
    "0011101", // 4
    "0111001", // 5
    "0000101", // 6
    "0010001", // 7
    "0001001", // 8
    "0010111"  // 9
};

static const char* const PATTERNS_C[10] = {
    "1110010", // 0
    "1100110", // 1
    "1101100", // 2
    "1000010", // 3
    "1011100", // 4
    "1001110", // 5
    "1010000", // 6
    "1000100", // 7
    "1001000", // 8
    "1110100"  // 9
};

static const int PARITY_TABLE[10][6] = {
    {0, 0, 0, 0, 0, 0}, // 0: AAAAAA
    {0, 0, 1, 0, 1, 1}, // 1: AABABB
    {0, 0, 1, 1, 0, 1}, // 2: AABBAB
    {0, 0, 1, 1, 1, 0}, // 3: AABBBA
    {0, 1, 0, 0, 1, 1}, // 4: ABAABB
    {0, 1, 1, 0, 0, 1}, // 5: ABBAAB
    {0, 1, 1, 1, 0, 0}, // 6: ABBBAA
    {0, 1, 0, 1, 0, 1}, // 7: ABABAB
    {0, 1, 0, 1, 1, 0}, // 8: ABABBA
    {0, 1, 1, 0, 1, 0}  // 9: ABBABA
};

static char computeCheckDigit(const std::string& payload)
{
    int sum = 0;
    int len = static_cast<int>(payload.length());
    for (int j = len - 1; j >= 0; --j)
    {
        int digit = payload[j] - '0';
        int distFromRight = (len - 1) - j;
        int weight = (distFromRight % 2 == 0) ? 3 : 1;
        sum += digit * weight;
    }
    int check = (10 - (sum % 10)) % 10;
    return static_cast<char>('0' + check);
}

BarcodeEncoder::BarcodeEncoder()
{
}

BarcodeEncoder::~BarcodeEncoder()
{
}

BarcodeEncoder::Params::Params()
{
    type = BarcodeEncoder::EAN_13;
    checksum_policy = BarcodeEncoder::APPEND_CHECKSUM;
    module_width = 2;
    bar_height = 50;
}

class BarcodeEncoderImpl : public BarcodeEncoder
{
public:
    BarcodeEncoderImpl(const BarcodeEncoder::Params& parameters) : params_(parameters)
    {
    }

    void encode(const String& encoded_info, OutputArray output) CV_OVERRIDE;

private:
    BarcodeEncoder::Params params_;
};

void BarcodeEncoderImpl::encode(const String& encoded_info, OutputArray output)
{
    if (params_.module_width <= 0)
    {
        CV_Error(Error::StsBadArg, "module_width must be greater than 0.");
    }
    if (params_.bar_height <= 0)
    {
        CV_Error(Error::StsBadArg, "bar_height must be greater than 0.");
    }

    if (encoded_info.empty())
    {
        CV_Error(Error::StsBadArg, "Input barcode text cannot be empty.");
    }

    for (size_t i = 0; i < encoded_info.length(); ++i)
    {
        char c = encoded_info[i];
        if (c < '0' || c > '9')
        {
            CV_Error(Error::StsBadArg, cv::format("Input character '%c' is not a digit. Only digits (0-9) are allowed.", c));
        }
    }

    std::string full_code;

    if (params_.checksum_policy == BarcodeEncoder::APPEND_CHECKSUM)
    {
        if (params_.type == BarcodeEncoder::EAN_13)
        {
            if (encoded_info.length() != 12)
            {
                CV_Error(Error::StsBadArg, cv::format("EAN-13 with APPEND_CHECKSUM expects 12 digits, got %zu.", encoded_info.length()));
            }
            char cd = computeCheckDigit(encoded_info);
            full_code = encoded_info + cd;
        }
        else if (params_.type == BarcodeEncoder::EAN_8)
        {
            if (encoded_info.length() != 7)
            {
                CV_Error(Error::StsBadArg, cv::format("EAN-8 with APPEND_CHECKSUM expects 7 digits, got %zu.", encoded_info.length()));
            }
            char cd = computeCheckDigit(encoded_info);
            full_code = encoded_info + cd;
        }
        else if (params_.type == BarcodeEncoder::UPC_A)
        {
            if (encoded_info.length() != 11)
            {
                CV_Error(Error::StsBadArg, cv::format("UPC-A with APPEND_CHECKSUM expects 11 digits, got %zu.", encoded_info.length()));
            }
            char cd = computeCheckDigit(encoded_info);
            full_code = encoded_info + cd;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported barcode type.");
        }
    }
    else if (params_.checksum_policy == BarcodeEncoder::VALIDATE_CHECKSUM)
    {
        if (params_.type == BarcodeEncoder::EAN_13)
        {
            if (encoded_info.length() != 13)
            {
                CV_Error(Error::StsBadArg, cv::format("EAN-13 with VALIDATE_CHECKSUM expects 13 digits, got %zu.", encoded_info.length()));
            }
            char expected_cd = computeCheckDigit(encoded_info.substr(0, 12));
            if (encoded_info[12] != expected_cd)
            {
                CV_Error(Error::StsBadArg, cv::format("Invalid check digit in EAN-13: expected '%c', got '%c'.", expected_cd, encoded_info[12]));
            }
            full_code = encoded_info;
        }
        else if (params_.type == BarcodeEncoder::EAN_8)
        {
            if (encoded_info.length() != 8)
            {
                CV_Error(Error::StsBadArg, cv::format("EAN-8 with VALIDATE_CHECKSUM expects 8 digits, got %zu.", encoded_info.length()));
            }
            char expected_cd = computeCheckDigit(encoded_info.substr(0, 7));
            if (encoded_info[7] != expected_cd)
            {
                CV_Error(Error::StsBadArg, cv::format("Invalid check digit in EAN-8: expected '%c', got '%c'.", expected_cd, encoded_info[7]));
            }
            full_code = encoded_info;
        }
        else if (params_.type == BarcodeEncoder::UPC_A)
        {
            if (encoded_info.length() != 12)
            {
                CV_Error(Error::StsBadArg, cv::format("UPC-A with VALIDATE_CHECKSUM expects 12 digits, got %zu.", encoded_info.length()));
            }
            char expected_cd = computeCheckDigit(encoded_info.substr(0, 11));
            if (encoded_info[11] != expected_cd)
            {
                CV_Error(Error::StsBadArg, cv::format("Invalid check digit in UPC-A: expected '%c', got '%c'.", expected_cd, encoded_info[11]));
            }
            full_code = encoded_info;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported barcode type.");
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Unsupported checksum policy.");
    }

    std::string bit_string;
    int left_quiet = 0;
    int right_quiet = 0;

    if (params_.type == BarcodeEncoder::EAN_13)
    {
        left_quiet = 11;
        right_quiet = 7;
        int first_digit = full_code[0] - '0';
        bit_string += "101"; // start guard pattern
        for (int i = 0; i < 6; ++i)
        {
            int d = full_code[i + 1] - '0';
            if (PARITY_TABLE[first_digit][i] == 0)
            {
                bit_string += PATTERNS_A[d];
            }
            else
            {
                bit_string += PATTERNS_B[d];
            }
        }
        bit_string += "01010"; // center guard pattern
        for (int i = 0; i < 6; ++i)
        {
            int d = full_code[i + 7] - '0';
            bit_string += PATTERNS_C[d];
        }
        bit_string += "101"; // stop guard pattern
    }
    else if (params_.type == BarcodeEncoder::UPC_A)
    {
        left_quiet = 9;
        right_quiet = 9;
        bit_string += "101"; // start guard pattern
        for (int i = 0; i < 6; ++i)
        {
            int d = full_code[i] - '0';
            bit_string += PATTERNS_A[d];
        }
        bit_string += "01010"; // center guard pattern
        for (int i = 0; i < 6; ++i)
        {
            int d = full_code[i + 6] - '0';
            bit_string += PATTERNS_C[d];
        }
        bit_string += "101"; // stop guard pattern
    }
    else if (params_.type == BarcodeEncoder::EAN_8)
    {
        left_quiet = 7;
        right_quiet = 7;
        bit_string += "101"; // start guard pattern
        for (int i = 0; i < 4; ++i)
        {
            int d = full_code[i] - '0';
            bit_string += PATTERNS_A[d];
        }
        bit_string += "01010"; // center guard pattern
        for (int i = 0; i < 4; ++i)
        {
            int d = full_code[i + 4] - '0';
            bit_string += PATTERNS_C[d];
        }
        bit_string += "101"; // stop guard pattern
    }

    int total_modules = left_quiet + static_cast<int>(bit_string.length()) + right_quiet;
    if (total_modules <= 0 || params_.module_width > INT_MAX / total_modules)
    {
        CV_Error(Error::StsBadArg, cv::format("module_width (%d) is too large for %d total modules and would cause integer overflow.",
                                              params_.module_width, total_modules));
    }
    int total_width = total_modules * params_.module_width;

    output.create(params_.bar_height, total_width, CV_8UC1);
    Mat out = output.getMat();
    out.setTo(Scalar(255)); // white background

    for (size_t m = 0; m < bit_string.length(); ++m)
    {
        if (bit_string[m] == '1')
        {
            int module_idx = left_quiet + static_cast<int>(m);
            int start_col = module_idx * params_.module_width;
            int end_col = (module_idx + 1) * params_.module_width;
            out.colRange(start_col, end_col).setTo(Scalar(0)); // black bar
        }
    }
}

Ptr<BarcodeEncoder> BarcodeEncoder::create(const BarcodeEncoder::Params& parameters)
{
    return makePtr<BarcodeEncoderImpl>(parameters);
}

}} // namespace cv::barcode

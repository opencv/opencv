// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Intel Corporation, all rights reserved.

#include "precomp.hpp"
#include "qrcode_encoder_table.inl.hpp"
#include "graphical_code_detector_impl.hpp"

namespace cv
{
using std::vector;

const int MAX_PAYLOAD_LEN = 8896;
const int MAX_FORMAT_LENGTH = 15;
const int MAX_VERSION_LENGTH = 18;
const int MODE_BITS_NUM = 4;
const uint8_t INVALID_REGION_VALUE = 110;

static void decToBin(const int dec_number, const int total_bits, std::vector<uint8_t> &bin_number);
static uint8_t gfPow(uint8_t x, int power);
static uint8_t gfMul(const uint8_t x, const uint8_t y);
static uint8_t gfDiv(const uint8_t x, const uint8_t y);
static void gfPolyMul(const vector<uint8_t> &p, const vector<uint8_t> &q, vector<uint8_t> &product);
static void gfPolyDiv(const vector<uint8_t> &dividend, const vector<uint8_t> &divisor, const int ecc_num, vector<uint8_t> &quotient);
static void polyGenerator(const int n, vector<uint8_t> &result);
static int getBits(const int bits, const vector<uint8_t> &payload, int &pay_index);

static void decToBin(const int dec_number, const int total_bits, std::vector<uint8_t> &bin_number)
{
    for (int i = 0; i < total_bits; i++)
    {
        bin_number[total_bits - i - 1] = (dec_number >> i) % 2;
    }
}

static void writeDecNumber(const int dec_number, const int total_bits, vector<uint8_t> &output_bits)
{
    std::vector<uint8_t> bin_number(total_bits);
    decToBin(dec_number, total_bits, bin_number);
    output_bits.insert(output_bits.end(), bin_number.begin(), bin_number.end());
}

static uint8_t gfPow(uint8_t x, int power)
{
    return gf_exp[(gf_log[x] * power) % 255];
}

static uint8_t gfMul(const uint8_t x, const uint8_t y)
{
    if (x == 0 || y == 0)
        return 0;
    return gf_exp[(gf_log[x] + gf_log[y]) % 255];
}

static uint8_t gfDiv(const uint8_t x, const uint8_t y)
{
    if (x == 0 || y == 0)
        return 0;
    return gf_exp[(gf_log[x] + 255 - gf_log[y]) % 255];
}

static void gfPolyMul(const vector<uint8_t> &p, const vector<uint8_t> &q, vector<uint8_t> &product)
{
    int len_p = (int)p.size();
    int len_q = (int)q.size();
    vector<uint8_t> temp_result(len_p + len_q - 1, 0);

    for (int j = 0; j < len_q; j++)
    {
        uint8_t q_val = q[j];
        if (!q_val)
            continue;
        for (int i = 0; i < len_p; i++)
        {
            uint8_t p_val = p[i];
            if (!p_val)
                continue;
            temp_result[i + j] ^= gfMul(p_val, q_val);
        }
    }
    product = temp_result;
}

static void gfPolyDiv(const vector<uint8_t> &dividend, const vector<uint8_t> &divisor, const int bits_needed, vector<uint8_t> &quotient)
{
    int dividend_len = (int)dividend.size() - 1;
    int divisor_len = (int)divisor.size() - 1;
    vector<uint8_t> temp = dividend;

    int times = dividend_len - divisor_len + 1;
    for (int i = 0; i < times; i++)
    {
        uint8_t dividend_val = temp[dividend_len - i];
        if(dividend_val != 0)
        {
            for (int j = 0; j < divisor_len + 1; j++)
            {
                uint8_t divisor_val = divisor[divisor_len - j];
                if (divisor_val != 0)
                {
                    temp[dividend_len - i - j] ^= gfMul(divisor_val, dividend_val);
                }
            }
        }
    }
    quotient = vector<uint8_t>(temp.begin(), temp.begin() + bits_needed);
}

static void polyGenerator(const int n, vector<uint8_t> &result)
{
    vector<uint8_t> temp(2, 1);
    result = vector<uint8_t>(1, 1);
    for (int i = 1; i <= n; i++)
    {
        temp[0] = gfPow(2, i - 1);
        gfPolyMul(result, temp, result);
    }
}

static int getBits(const int bits, const vector<uint8_t> &payload, int &pay_index)
{
    int result = 0;
    for (int i = 0; i < bits; i++)
    {
        result = result << 1;
        result += payload[pay_index++];
    }
    return result;
}

static int mapSymbol(char c)
{
    if (c >= '0' && c <= '9')
        return (int)(c - '0');
    if (c >= 'A' && c <= 'Z')
        return (int)(c - 'A') + 10;
    switch (c)
    {
        case ' ': return 36 + 0;
        case '$': return 36 + 1;
        case '%': return 36 + 2;
        case '*': return 36 + 3;
        case '+': return 36 + 4;
        case '-': return 36 + 5;
        case '.': return 36 + 6;
        case '/': return 36 + 7;
        case ':': return 36 + 8;
    }
    return -1;
}

static void maskData(const Mat& original, const int mask_type_num, Mat &masked);

QRCodeEncoder::QRCodeEncoder()
{
    // nothing
}

QRCodeEncoder::~QRCodeEncoder()
{
    // nothing
}

QRCodeEncoder::Params::Params()
{
    version = 0;
    correction_level = CORRECT_LEVEL_L;
    mode = MODE_AUTO;
    structure_number = 1;
}

class QRCodeEncoderImpl : public QRCodeEncoder
{
public:
    QRCodeEncoderImpl(const QRCodeEncoder::Params& parameters) : params(parameters)
    {
        version_level = parameters.version;
        ecc_level = parameters.correction_level;
        mode_type = parameters.mode;
        struct_num = parameters.structure_number;
        version_size = 21;
        mask_type = 0;
        parity = 0;
        sequence_num = 0;
        total_num = 0;
    }

    void encode(const String& encoded_info, OutputArray qrcode) CV_OVERRIDE;
    void encodeStructuredAppend(const String& input, OutputArrayOfArrays output) CV_OVERRIDE;
    QRCodeEncoder::Params params;
protected:
    int version_level;
    CorrectionLevel ecc_level;
    EncodeMode mode_type;
    int struct_num;
    int version_size;
    int mask_type;
    vector<uint8_t> format;
    vector<uint8_t> version_reserved;
    vector<uint8_t>	payload;
    vector<uint8_t>	rearranged_data;
    Mat original;
    Mat masked_data;
    uint8_t parity;
    uint8_t sequence_num;
    uint8_t total_num;
    vector<Mat> final_qrcodes;

    const VersionInfo* version_info;
    const BlockParams* cur_ecc_params;

    bool isNumeric(const std::string& input) const;
    bool isAlphaNumeric(const std::string& input) const;
    EncodeMode autoEncodeMode(const std::string &input) const ;
    bool encodeByte(const std::string& input, vector<uint8_t> &output);
    bool encodeAlpha(const std::string& input, vector<uint8_t> &output);
    bool encodeNumeric(const std::string& input, vector<uint8_t> &output);
    bool encodeECI(const std::string& input, vector<uint8_t> &output);
    bool encodeKanji(const std::string& input, vector<uint8_t> &output);
    bool encodeAuto(const std::string& input, vector<uint8_t> &output, EncodeMode *mode = nullptr);
    bool encodeStructure(const std::string& input, vector<uint8_t> &output);
    int eccLevelToCode(CorrectionLevel level);
    void padBitStream();
    bool stringToBits(const std::string& input_info);
    void eccGenerate(vector<vector<uint8_t> > &data_blocks, vector<vector<uint8_t> > &ecc_blocks);
    void rearrangeBlocks(const vector<vector<uint8_t> > &data_blocks, const vector<vector<uint8_t> > &ecc_blocks);
    void writeReservedArea();
    bool writeBit(int x, int y, bool value);
    void writeData();
    void structureFinalMessage();
    void formatGenerate(const int mask_type_num, vector<uint8_t> &format_array);
    void versionInfoGenerate(const int version_level_num, vector<uint8_t> &version_array);
    void fillReserved(const vector<uint8_t> &format_array, Mat &masked);
    void findAutoMaskType();
    bool estimateVersion(const int input_length, EncodeMode mode, vector<int> &possible_version);
    int versionAuto(const std::string &input_str);
    int findVersionCapacity(const int input_length, const int ecc, const std::vector<int>& possible_versions);
    void generatingProcess(const std::string& input, Mat &qrcode);
    void generateQR(const std::string& input);
};

int QRCodeEncoderImpl::eccLevelToCode(CorrectionLevel level)
{
    switch (level)
    {
        case CORRECT_LEVEL_L:
            return 0b01;
        case CORRECT_LEVEL_M:
            return 0b00;
        case CORRECT_LEVEL_Q:
            return 0b11;
        case CORRECT_LEVEL_H:
            return 0b10;
    }
    CV_Error( Error::StsBadArg,
        "Error correction level is incorrect. Available levels are"
        "CORRECT_LEVEL_L, CORRECT_LEVEL_M, CORRECT_LEVEL_Q, CORRECT_LEVEL_H." );
}

int QRCodeEncoderImpl::findVersionCapacity(const int input_length, const int ecc, const std::vector<int>& possible_versions)
{
    int data_codewords, version_index = -1;
    const int byte_len = 8;
    version_index = -1;

    for (int i : possible_versions)
    {
        auto& tmp_ecc_params = version_info_database[i].ecc[ecc];
        data_codewords = tmp_ecc_params.data_codewords_in_G1 * tmp_ecc_params.num_blocks_in_G1 +
                         tmp_ecc_params.data_codewords_in_G2 * tmp_ecc_params.num_blocks_in_G2;

        if (data_codewords * byte_len >= input_length)
        {
            version_index = i;
            break;
        }
    }
    return version_index;
}

static inline int getCapacity(int version, QRCodeEncoder::CorrectionLevel ecc_level, QRCodeEncoder::EncodeMode mode) {
    const int* capacity = version_capacity_database[version].ec_level[ecc_level].encoding_modes;
    switch (mode) {
        case QRCodeEncoder::EncodeMode::MODE_NUMERIC:
            return capacity[0];
        case QRCodeEncoder::EncodeMode::MODE_ALPHANUMERIC:
            return capacity[1];
        case QRCodeEncoder::EncodeMode::MODE_BYTE:
            return capacity[2];
        case QRCodeEncoder::EncodeMode::MODE_KANJI:
            return capacity[3];
        default:
            CV_Error(Error::StsNotImplemented, format("Unexpected mode %d", mode));
    }
}

bool QRCodeEncoderImpl::estimateVersion(const int bit_length, EncodeMode mode, vector<int>& possible_version)
{
    possible_version.clear();

    CV_Assert(mode != EncodeMode::MODE_AUTO);

    if (bit_length > getCapacity(MAX_VERSION, ecc_level, mode) * 8)
    {
        return false;
    }

    int version = MAX_VERSION;

    for (; version > 0; --version)
    {
        if (bit_length > getCapacity(version, ecc_level, mode)* 8) {
            break;
        }
    }

    if (version < MAX_VERSION)
    {
        version += 1;
    }

    possible_version.push_back(version);

    if (version < MAX_VERSION)
    {
        possible_version.push_back(version + 1);
    }

    return true;
}

int QRCodeEncoderImpl::versionAuto(const std::string& input_str)
{
    vector<uint8_t> payload_tmp;
    EncodeMode mode;
    encodeAuto(input_str, payload_tmp, &mode);

    vector<int> possible_version;
    int bit_length = static_cast<int>(payload_tmp.size());
    if (!estimateVersion(bit_length, mode, possible_version)) {
        return -1;
    }

    // Extra info for structure's position, total and parity + mode of final message
    if (mode_type == MODE_STRUCTURED_APPEND)
        bit_length += 4 + 4 + 8 + 4;

    const auto tmp_version = findVersionCapacity(bit_length, ecc_level, possible_version);

    return tmp_version;
}

void QRCodeEncoderImpl::generateQR(const std::string &input)
{
    if (struct_num > 1)
    {
        for (size_t i = 0; i < input.length(); i++)
        {
            parity ^= input[i];
        }
        if (struct_num > 16)
        {
            struct_num = 16;
        }
        total_num = (uint8_t) struct_num - 1;
    }

    auto string_itr = input.begin();
    for (int i = struct_num; i > 0; --i)
    {
        sequence_num = (uint8_t) (struct_num - i);
        size_t segment_begin = string_itr - input.begin();
        size_t segment_end = (input.end() - string_itr) / i;

        std::string input_info = input.substr(segment_begin, segment_end);
        string_itr += segment_end;

        int detected_version = versionAuto(input_info);
        int tmp_version_level = version_level;
        if (detected_version == -1)
            CV_Error(Error::StsBadArg, "The given input exceeds the maximum capacity of a QR code with the selected encoding mode and error correction level " );
        else if (tmp_version_level == 0)
            tmp_version_level = detected_version;
        else if (tmp_version_level < detected_version)
            CV_Error(Error::StsBadArg, "The given version is not suitable for the given input string length ");

        payload.clear();
        payload.reserve(MAX_PAYLOAD_LEN);
        format = vector<uint8_t> (15, 255);
        version_reserved = vector<uint8_t> (18, 255);
        version_size = (21 + (tmp_version_level - 1) * 4);
        version_info = &version_info_database[tmp_version_level];
        cur_ecc_params = &version_info->ecc[ecc_level];
        original = Mat(Size(version_size, version_size), CV_8UC1, Scalar(255));
        masked_data = original.clone();
        Mat qrcode = masked_data.clone();
        generatingProcess(input_info, qrcode);
        final_qrcodes.push_back(qrcode);
    }
}

void QRCodeEncoderImpl::formatGenerate(const int mask_type_num, vector<uint8_t> &format_array)
{
    int idx = (eccLevelToCode(ecc_level) << 3) | mask_type_num;
    format_array.resize(MAX_FORMAT_LENGTH);
    for (int i = 0; i < MAX_FORMAT_LENGTH; ++i) {
        format_array[i] = (formatInfoLUT[idx] >> i) & 1;
    }
}

void QRCodeEncoderImpl::versionInfoGenerate(const int version_level_num, vector<uint8_t> &version_array)
{
    const int version_bits_num = 6;
    std::vector<uint8_t> version_bits(version_bits_num);
    decToBin(version_level_num, version_bits_num, version_bits);

    std::reverse(version_bits.begin(), version_bits.end());
    vector<uint8_t> shift(12, 0);
    vector<uint8_t> polynomial;
    hconcat(shift, version_bits, polynomial);

    const int generator_len = 13;
    const uint8_t generator_arr[generator_len] = {1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1};
    std::vector<uint8_t> format_mask (generator_arr, generator_arr + sizeof(generator_arr) / sizeof(generator_arr[0]));

    vector<uint8_t> ecc_code;
    gfPolyDiv(polynomial, format_mask, 12, ecc_code);
    hconcat(ecc_code, version_bits, version_array);
}

bool QRCodeEncoderImpl::encodeAlpha(const std::string& input, vector<uint8_t>& output)
{
    writeDecNumber(MODE_ALPHANUMERIC, MODE_BITS_NUM, output);

    int length_bits_num = 13;
    if (version_level < 10)
        length_bits_num = 9;
    else if (version_level < 27)
        length_bits_num = 11;

    int str_len = int(input.length());
    writeDecNumber(str_len, length_bits_num, output);

    const int alpha_symbol_bits = 11;
    const int residual_bits = 6;
    for (int i = 0; i < str_len - 1; i += 2)
    {
        int index_1 = mapSymbol(input[i]);
        int index_2 = mapSymbol(input[i + 1]);

        if(index_1 == -1 || (index_2 == -1 && i + 1 < str_len))
            return false;
        int alpha = index_1 * 45 + index_2;

        writeDecNumber(alpha, alpha_symbol_bits, output);
    }
    if (str_len % 2 != 0)
    {
        int index_residual_elem = mapSymbol(*input.rbegin());
        if(index_residual_elem == -1)
            return false;

        writeDecNumber(index_residual_elem, residual_bits, output);
    }
    return true;
}

bool QRCodeEncoderImpl::encodeByte(const std::string& input, vector<uint8_t>& output)
{
    writeDecNumber(MODE_BYTE, MODE_BITS_NUM, output);

    int length_bits_num = 8;
    if (version_level > 9)
        length_bits_num = 16;

    int str_len = int(input.length());
    writeDecNumber(str_len, length_bits_num, output);

    const int byte_symbol_bits = 8;
    for (int i = 0; i < str_len; i++)
    {
        writeDecNumber(uint8_t(input[i]), byte_symbol_bits, output);
    }
    return true;
}

bool QRCodeEncoderImpl::encodeNumeric(const std::string& input,vector<uint8_t>& output)
{
    writeDecNumber(MODE_NUMERIC, MODE_BITS_NUM, output);

    int length_bits_num = 10;
    if (version_level >= 27)
        length_bits_num = 14;
    else if (version_level >= 10)
        length_bits_num = 12;

    int str_len = int(input.length());
    writeDecNumber(str_len, length_bits_num, output);

    int count = 0;
    const int num_symbol_bits = 10;
    const int residual_bits[2] = {7, 4};
    while (count + 3 <= str_len)
    {
        if (input[count] > '9' || input[count] < '0' ||
            input[count + 1] > '9' || input[count + 1] < '0' ||
            input[count + 2] > '9' || input[count + 2] < '0')
            return false;
        int num = 100 * (int)(input[count] - '0') +
                  10 * (int)(input[count + 1] - '0') +
                  (int)(input[count + 2] - '0');

        writeDecNumber(num, num_symbol_bits, output);
        count += 3;
    }
    if (count + 2 == str_len)
    {
        if (input[count] > '9' || input[count] < '0'||
            input[count + 1] >'9'|| input[count + 1] < '0')
            return false;
        int num = 10 * (int)(input[count] - '0') +
                  (int)(input[count + 1] - '0');

        writeDecNumber(num, residual_bits[0], output);
    }
    else if (count + 1 == str_len)
    {
        if (input[count] > '9' || input[count] < '0')
            return false;
        int num = (int)(input[count] - '0');

        writeDecNumber(num, residual_bits[1], output);
    }
    return true;
}

bool QRCodeEncoderImpl::encodeECI(const std::string& input, vector<uint8_t>& output)
{
    writeDecNumber(MODE_ECI, MODE_BITS_NUM, output);
    const uint32_t assign_value_range[3] = {127, 16383, 999999};

    // by adding other ECI modes `eci_assignment_number` can be moved to algorithm parameters
    uint32_t eci_assignment_number = ECI_UTF8; // utf-8

    int codewords = 1;
    if(eci_assignment_number > assign_value_range[2])
        return false;
    if (eci_assignment_number > assign_value_range[1])
        codewords = 3;
    else if (eci_assignment_number > assign_value_range[0])
        codewords = 2;

    const int bits = 8;
    switch (codewords)
    {
        case 1:
            writeDecNumber(0, codewords, output);
            writeDecNumber(eci_assignment_number, codewords * bits - 1, output);
            break;
        case 2:
            writeDecNumber(2, codewords, output);
            writeDecNumber(eci_assignment_number, codewords * bits - 2, output);
            break;
        case 3:
            writeDecNumber(6, codewords, output);
            writeDecNumber(eci_assignment_number, codewords * bits - 3, output);
            break;
    }

    encodeByte(input, output);
    return true;
}

bool QRCodeEncoderImpl::encodeKanji(const std::string& input, vector<uint8_t>& output)
{
    writeDecNumber(MODE_KANJI, MODE_BITS_NUM, output);

    int length_bits_num = 8;
    if (version_level >= 10)
        length_bits_num = 10;
    else if (version_level >= 27)
        length_bits_num = 12;

    int str_len = int(input.length()) / 2;
    writeDecNumber(str_len, length_bits_num, output);

    const int kanji_symbol_bits = 13;
    int i = 0;
    while(i < str_len * 2)
    {
        uint16_t high_byte = (uint16_t)(input[i] & 0xff);
        uint16_t low_byte = (uint16_t)(input[i+1] & 0xff);
        uint16_t per_char = (high_byte << 8) + (low_byte);

        if(0x8140 <= per_char && per_char <= 0x9FFC)
        {
            per_char -= 0x8140;
        }
        else if(0xE040 <= per_char && per_char <= 0xEBBF)
        {
            per_char -= 0xC140;
        }
        uint16_t new_high = per_char >> 8;
        uint16_t result = new_high * 0xC0;
        result += (per_char & 0xFF);

        writeDecNumber(result, kanji_symbol_bits, output);
        i += 2;
    }
    return true;
}

bool QRCodeEncoderImpl::encodeStructure(const std::string& input, vector<uint8_t>& output)
{
    const int num_field = 4;
    const int checksum_field = 8;
    writeDecNumber(MODE_STRUCTURED_APPEND, MODE_BITS_NUM, output);
    writeDecNumber(sequence_num, num_field, output);
    writeDecNumber(total_num, num_field, output);
    writeDecNumber(parity, checksum_field, output);

    return encodeAuto(input, output);
}

bool QRCodeEncoderImpl::isNumeric(const std::string& input) const
{
    for (size_t i = 0; i < input.length(); i++)
    {
        if (input[i] < '0' || input[i] > '9')
            return false;
    }
    return true;
}

bool QRCodeEncoderImpl::isAlphaNumeric(const std::string& input) const
{
    for (size_t i = 0; i < input.length(); i++)
    {
        if (mapSymbol(input[i]) == -1)
            return false;
    }
    return true;
}

QRCodeEncoder::EncodeMode QRCodeEncoderImpl::autoEncodeMode(const std::string &input) const
{
    if (isNumeric(input))
    {
        return EncodeMode::MODE_NUMERIC;
    }

    if (isAlphaNumeric(input))
    {
        return EncodeMode::MODE_ALPHANUMERIC;
    }

    return EncodeMode::MODE_BYTE;
}

bool QRCodeEncoderImpl::encodeAuto(const std::string& input, vector<uint8_t>& output, EncodeMode *mode)
{
    const auto selected_mode = autoEncodeMode(input);

    CV_Assert(selected_mode != EncodeMode::MODE_AUTO);

    switch (selected_mode)
    {
        case EncodeMode::MODE_NUMERIC:
            encodeNumeric(input, output);
            break;
        case EncodeMode::MODE_ALPHANUMERIC:
            encodeAlpha(input, output);
            break;
        case EncodeMode::MODE_STRUCTURED_APPEND:
            encodeByte(input, output);
            break;
        case EncodeMode::MODE_BYTE:
            encodeByte(input, output);
            break;
        case EncodeMode::MODE_KANJI:
            encodeKanji(input, output);
            break;
        case EncodeMode::MODE_ECI:
            encodeECI(input, output);
            break;
        default:
            break;
    }

    if (mode != nullptr)
    {
        *mode = selected_mode;
    }

    return true;
}

void QRCodeEncoderImpl::padBitStream()
{
    int total_data = version_info->total_codewords -
                     cur_ecc_params->ecc_codewords * (cur_ecc_params->num_blocks_in_G1 + cur_ecc_params->num_blocks_in_G2);
    const int bits = 8;
    total_data *= bits;
    int pad_num = total_data - (int)payload.size();

    if (pad_num <= 0)
        return;
    else if (pad_num <= 4)
    {
        int payload_size = (int)payload.size();
        payload.insert(payload.end(), payload_size, 0);
    }
    else
    {
        payload.insert(payload.end(), 4, 0);

        int i = payload.size() % bits;

        if (i != 0)
        {
            payload.insert(payload.end(), bits - i, 0);
        }
        pad_num = total_data - (int)payload.size();

        if (pad_num > 0)
        {
            const int pad_patterns[2] = {236, 17};
            int num = pad_num / bits;
            const int pattern_size = 8;
            for (int j = 0; j < num; j++)
            {
                writeDecNumber(pad_patterns[j % 2], pattern_size, payload);
            }
        }
    }
}

bool QRCodeEncoderImpl::stringToBits(const std::string& input_info)
{
    switch (mode_type)
    {
        case MODE_NUMERIC:
            return encodeNumeric(input_info, payload);
        case MODE_ALPHANUMERIC:
            return encodeAlpha(input_info, payload);
        case MODE_STRUCTURED_APPEND:
            return encodeStructure(input_info, payload);
        case MODE_BYTE:
            return encodeByte(input_info, payload);
        case MODE_ECI:
            return encodeECI(input_info, payload);
        case MODE_KANJI:
            return encodeKanji(input_info, payload);
        default:
            return encodeAuto(input_info, payload);
    }
}

void QRCodeEncoderImpl::eccGenerate(vector<vector<uint8_t> > &data_blocks, vector<vector<uint8_t> > &ecc_blocks)
{
    const int ec_codewords = cur_ecc_params->ecc_codewords;
    int pay_index = 0;
    vector<uint8_t> g_x;
    polyGenerator(ec_codewords, g_x);
    int blocks = cur_ecc_params->num_blocks_in_G2 + cur_ecc_params->num_blocks_in_G1;
    for (int i = 0; i < blocks; i++)
    {
        int block_len = 0;

        if (i < cur_ecc_params->num_blocks_in_G1)
        {
            block_len = cur_ecc_params->data_codewords_in_G1;
        }
        else
        {
            block_len = cur_ecc_params->data_codewords_in_G2;
        }
        vector<uint8_t> block_i (block_len, 0);

        for (int j = 0; j < block_len; j++)
        {
            block_i[block_len - 1 - j] = (uchar)getBits(8, payload, pay_index);
        }
        vector<uint8_t> dividend;
        vector<uint8_t> shift (ec_codewords, 0);
        hconcat(shift, block_i, dividend);
        vector<uint8_t> ecc_i;
        gfPolyDiv(dividend, g_x, ec_codewords, ecc_i);
        data_blocks.push_back(block_i);
        ecc_blocks.push_back(ecc_i);
    }
}

void QRCodeEncoderImpl::rearrangeBlocks(const vector<vector<uint8_t> > &data_blocks, const vector<vector<uint8_t> > &ecc_blocks)
{
    rearranged_data.clear();
    int blocks = cur_ecc_params->num_blocks_in_G2 + cur_ecc_params->num_blocks_in_G1;
    int col_border = max(cur_ecc_params->data_codewords_in_G2, cur_ecc_params->data_codewords_in_G1);
    int total_codeword_num = version_info->total_codewords;
    int is_not_equal = cur_ecc_params->data_codewords_in_G2 - cur_ecc_params->data_codewords_in_G1;
    int add_steps = cur_ecc_params->data_codewords_in_G2 > cur_ecc_params->data_codewords_in_G1 ?
                   (cur_ecc_params->data_codewords_in_G2 - cur_ecc_params->data_codewords_in_G1) * cur_ecc_params->num_blocks_in_G1 : 0;
    rearranged_data.reserve(total_codeword_num + add_steps);
    for (int i = 0; i < total_codeword_num + add_steps; i++)
    {
        int cur_col = i / blocks;
        int cur_row = i % blocks;
        int data_col = (int)data_blocks[cur_row].size() - 1;
        int ecc_col = (int)ecc_blocks[cur_row].size() - 1;
        uint8_t tmp = 0;

        if (cur_col < col_border)
        {
            if (is_not_equal && cur_col == cur_ecc_params->data_codewords_in_G2 - 1 && cur_row < cur_ecc_params->num_blocks_in_G1)
            {
                continue;
            }
            else
            {
                tmp = data_blocks[cur_row][data_col - cur_col];
            }
        }
        else
        {
            int index = ecc_col - (cur_col - col_border);
            tmp = ecc_blocks[cur_row][index];
        }
        rearranged_data.push_back(tmp);
    }
}

void QRCodeEncoderImpl::findAutoMaskType()
{
    int best_index = 0;
    int lowest_penalty = INT_MAX;
    int penalty_two_value = 3, penalty_three_value = 40;
    for (int cur_type = 0; cur_type < 8; cur_type++)
    {
        Mat test_result = masked_data.clone();
        vector<uint8_t> test_format = format;
        maskData(original, cur_type, test_result);
        formatGenerate(cur_type, test_format);
        fillReserved(test_format, test_result);
        int continued_num = 0;
        int penalty_one = 0, penalty_two = 0, penalty_three = 0, penalty_four = 0, penalty_total = 0;
        int current_color = -1;

        for (int direction = 0; direction < 2; direction++)
        {
            if (direction != 0)
            {
                test_result = test_result.t();
            }
            for (int i = 0; i < version_size; i++)
            {
                int per_row = 0;
                for (int j = 0; j < version_size; j++)
                {
                    if (j == 0)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        continued_num = 1;
                        continue;
                    }
                    if (current_color == test_result.at<uint8_t>(i, j))
                    {
                        continued_num += 1;
                    }
                    if (current_color != test_result.at<uint8_t>(i, j) || j + 1 == version_size)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        if (continued_num >= 5)
                        {
                            per_row += 3 + continued_num - 5;
                        }
                        continued_num = 1;
                    }
                }
                penalty_one += per_row;
            }
        }
        for (int i = 0; i < version_size - 1; i++)
        {
            for (int j = 0; j < version_size - 1; j++)
            {
                uint8_t color = test_result.at<uint8_t>(i, j);
                if (color == test_result.at<uint8_t>(i, j + 1) &&
                    color == test_result.at<uint8_t>(i + 1, j + 1) &&
                    color == test_result.at<uint8_t>(i + 1, j))
                {
                    penalty_two += penalty_two_value;
                }
            }
        }
        Mat penalty_pattern[2];
        penalty_pattern[0] = (Mat_<uint8_t >(1, 11) << 255, 255, 255, 255, 0, 255, 0, 0, 0, 255, 0);
        penalty_pattern[1] = (Mat_<uint8_t >(1, 11) << 0, 255, 0, 0, 0, 255, 0, 255, 255, 255, 255);
        for (int direction = 0; direction < 2; direction++)
        {
            if (direction != 0)
            {
                test_result = test_result.t();
            }
            for (int i = 0; i < version_size; i++)
            {
                int per_row = 0;
                for (int j = 0; j < version_size - 10; j++)
                {
                    Mat cur_test = test_result(Range(i, i + 1), Range(j, j + 11));
                    for (int pattern_index = 0; pattern_index < 2; pattern_index++)
                    {
                        Mat diff = (penalty_pattern[pattern_index] != cur_test);
                        bool equal = (countNonZero(diff) == 0);
                        if (equal)
                        {
                            per_row += penalty_three_value;
                        }
                    }
                }
                penalty_three += per_row;
            }
        }
        int dark_modules = 0;
        int total_modules = 0;
        for (int i = 0; i < version_size; i++)
        {
            for (int j = 0; j < version_size; j++)
            {
                if (test_result.at<uint8_t>(i, j) == 0)
                {
                    dark_modules += 1;
                }
                total_modules += 1;
            }
        }
        if (total_modules == 0)
            continue; // TODO: refactor, extract functions to reduce complexity
        int modules_percent = dark_modules * 100 / total_modules;
        int lower_bound = 45;
        int upper_bound = 55;
        int diff = min(abs(modules_percent - lower_bound), abs(modules_percent - upper_bound));
        penalty_four = (diff / 5) * 10;
        penalty_total = penalty_one + penalty_two + penalty_three + penalty_four;
        if (penalty_total < lowest_penalty)
        {
            best_index = cur_type;
            lowest_penalty = penalty_total;
        }
    }
    mask_type = best_index;
}

void maskData(const Mat& original, const int mask_type_num, Mat& masked)
{
    int version_size = original.rows;
    for (int i = 0; i < version_size; i++)
    {
        for (int j = 0; j < version_size; j++)
        {
            if (original.at<uint8_t>(i, j) == INVALID_REGION_VALUE)
            {
                continue;
            }
            else if((mask_type_num == 0 && !((i + j) % 2)) ||
                    (mask_type_num == 1 && !(i % 2)) ||
                    (mask_type_num == 2 && !(j % 3)) ||
                    (mask_type_num == 3 && !((i + j) % 3)) ||
                    (mask_type_num == 4 && !(((i / 2) + (j / 3)) % 2)) ||
                    (mask_type_num == 5 && !((i * j) % 2 + (i * j) % 3))||
                    (mask_type_num == 6 && !(((i * j) % 2 + (i * j) % 3) % 2))||
                    ((mask_type_num == 7 && !(((i * j) % 3 + (i + j) % 2) % 2))))
            {
                masked.at<uint8_t>(i, j) = original.at<uint8_t>(i, j) ^ 255;
            }
        }
    }
}

void QRCodeEncoderImpl::writeReservedArea()
{
    vector<Rect> finder_pattern(3);
    finder_pattern[0] = Rect(Point(0, 0), Point(9, 9));
    finder_pattern[1] = Rect(Point(0, (unsigned)version_size - 8), Point(9, version_size));
    finder_pattern[2] = Rect(Point((unsigned)version_size - 8, 0), Point(version_size, 9));
    const int coordinates_num = 2;
    int locator_position[coordinates_num] = {3, version_size - 1 - 3};

    for (int first_coordinate = 0; first_coordinate < coordinates_num; first_coordinate++)
    {
        for (int second_coordinate = 0; second_coordinate < coordinates_num; second_coordinate++)
        {
            if (first_coordinate == 1 && second_coordinate == 1)
            {
                continue;
            }
            int x = locator_position[first_coordinate];
            int y = locator_position[second_coordinate];
            for (int i = -5; i <= 5; i++)
            {
                for (int j = -5; j <= 5; j++)
                {
                    if (x + i < 0 || x + i >= version_size || y + j < 0 || y + j >= version_size)
                    {
                        continue;
                    }
                    if (!(abs(j) == 2 && abs(i) <= 2) &&
                        !(abs(j) <= 2 && abs(i) == 2) &&
                        !(abs(i) == 4) && !(abs(j) == 4))
                    {
                        masked_data.at<uint8_t>(x + i, y + j) = 0;
                    }
                    if ((y == locator_position[1] && j == -5) || (x == locator_position[1] && i == -5))
                    {
                        continue;
                    }
                    else
                    {
                        original.at<uint8_t>(x + i, y + j) = INVALID_REGION_VALUE;
                    }
                }
            }

        }
    }
    int x = locator_position[1] - 4;
    int y = locator_position[0] + 5;
    masked_data.at<uint8_t>(x, y) = 0;
    original.at<uint8_t>(x, y) = INVALID_REGION_VALUE;
    if (version_level >= 7)
    {
        for (int i = 0; i <= 5; i++)
        {
            for (int j = version_size - 11; j <= version_size - 8; j++)
            {
                original.at<uint8_t>(i, j) = INVALID_REGION_VALUE;
                original.at<uint8_t>(j, i) = INVALID_REGION_VALUE;
            }
        }
    }
    for (int i = 0; i < version_size; i++)
    {
        for (int j = 0; j < version_size; j++)
        {
            if (original.at<uint8_t>(i, j) == INVALID_REGION_VALUE)
            {
                continue;
            }
            if ((i == 6 || j == 6))
            {
                original.at<uint8_t>(i, j) = INVALID_REGION_VALUE;
                if (!((i == 6) && (j - 7) % 2 == 0) &&
                    !((j == 6) && ((i - 7) % 2 == 0)))
                {
                    masked_data.at<uint8_t>(i, j) = 0;
                }
            }
        }
    }
    for (int first_coord = 0; first_coord < MAX_ALIGNMENT && version_info->alignment_pattern[first_coord]; first_coord++)
    {
        for (int second_coord = 0; second_coord < MAX_ALIGNMENT && version_info->alignment_pattern[second_coord]; second_coord++)
        {
            x = version_info->alignment_pattern[first_coord];
            y = version_info->alignment_pattern[second_coord];
            bool is_in_finder = false;
            for (size_t i = 0; i < finder_pattern.size(); i++)
            {
                Rect rect = finder_pattern[i];
                if (x >= rect.tl().x && x <= rect.br().x
                    &&
                    y >= rect.tl().y && y <= rect.br().y)
                {
                    is_in_finder = true;
                    break;
                }
            }
            if (!is_in_finder)
            {
                for (int i = -2; i <= 2; i++)
                {
                    for (int j = -2; j <= 2; j++)
                    {
                        original.at<uint8_t>(x + i, y + j) = INVALID_REGION_VALUE;
                        if ((j == 0 && i == 0) || (abs(j) == 2) || abs(i) == 2)
                        {
                            masked_data.at<uint8_t>(x + i, y + j) = 0;
                        }
                    }
                }
            }
        }
    }
}

bool QRCodeEncoderImpl::writeBit(int x, int y, bool value)
{
    if (original.at<uint8_t>(y, x) == INVALID_REGION_VALUE)
    {
        return false;
    }
    if (!value)
    {
        original.at<uint8_t>(y, x) = 0;
        masked_data.at<uint8_t>(y, x) = 0;
    }
    original.at<uint8_t>(y, x) = static_cast<uint8_t>(255 * value);
    masked_data.at<uint8_t>(y, x) = static_cast<uint8_t>(255 * value);
    return true;
}

void QRCodeEncoderImpl::writeData()
{
    int y = version_size - 1;
    int x = version_size - 1;
    int dir = -1;
    int count = 0;
    int codeword_value = rearranged_data[0];
    const int limit_bits = (int)rearranged_data.size() * 8;
    bool limit_reached = false;
    while (x > 0)
    {
        if (x == 6)
        {
            x --;
        }
        for(int i = 0; i <= 1; i++)
        {
            bool bit_value = (codeword_value & (0x80 >> count % 8)) == 0;
            bool success = writeBit(x - i, y, bit_value);
            if (!success)
            {
                continue;
            }
            count++;
            if (count == limit_bits)
            {
                limit_reached = true;
                break;
            }
            if (count % 8 == 0)
            {
                codeword_value = rearranged_data[count / 8];
            }
        }
        if (limit_reached)
        {
            break;
        }
        y += dir;
        if (y < 0 || y >= version_size)
        {
            dir = -dir;
            x -= 2;
            y += dir;
        }
    }
}

void QRCodeEncoderImpl::fillReserved(const vector<uint8_t> &format_array, Mat &masked)
{
    for (int i = 0; i < 7; i++)
    {
        if (format_array[MAX_FORMAT_LENGTH - 1 - i] == 0)
        {
            masked.at<uint8_t>(version_size - 1 - i, 8) = 255;
        }
        else
        {
            masked.at<uint8_t>(version_size - 1 - i, 8) = 0;
        }
    }
    for (int i = 0; i < 8; i++)
    {
        if (format_array[MAX_FORMAT_LENGTH - 1 - (7 + i)] == 0)
        {
            masked.at<uint8_t>(8, version_size - 8 + i) = 255;
        }
        else
        {
            masked.at<uint8_t>(8, version_size - 8 + i) = 0;
        }
    }
    static const int xs_format[MAX_FORMAT_LENGTH] = {
            8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
    };
    static const int ys_format[MAX_FORMAT_LENGTH] = {
            0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
    };
    for (int i = MAX_FORMAT_LENGTH - 1; i >= 0; i--)
    {
        if (format_array[i] == 0)
        {
            masked.at<uint8_t>(ys_format[i], xs_format[i]) = 255;
        }
        else
        {
            masked.at<uint8_t>(ys_format[i], xs_format[i]) = 0;
        }
    }

    if (version_level >= 7)
    {
        const int max_size = version_size;
        const int version_block_width = 2;
        const int xs_version[version_block_width][MAX_VERSION_LENGTH] = {
                { 5, 5, 5,
                  4, 4, 4,
                  3, 3, 3,
                  2, 2, 2,
                  1, 1, 1,
                  0, 0, 0
                },
                { max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11
                }
        };
        const int ys_version[version_block_width][MAX_VERSION_LENGTH] = {
                { max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11,
                  max_size - 9, max_size - 10, max_size - 11
                },
                { 5, 5, 5,
                  4, 4, 4,
                  3, 3, 3,
                  2, 2, 2,
                  1, 1, 1,
                  0, 0, 0,
                }
        };
        for (int i = 0; i < version_block_width; i++)
        {
            for (int j = 0; j < MAX_VERSION_LENGTH; j++)
            {
                if (version_reserved[MAX_VERSION_LENGTH - j - 1] == 0)
                {
                    masked.at<uint8_t>(ys_version[i][j], xs_version[i][j]) = 255;
                }
                else
                {
                    masked.at<uint8_t>(ys_version[i][j], xs_version[i][j]) = 0;
                }
            }
        }
    }
}

void QRCodeEncoderImpl::structureFinalMessage()
{
    writeReservedArea();
    writeData();
    findAutoMaskType();
    maskData(original, mask_type, masked_data);
    formatGenerate(mask_type, format);
    versionInfoGenerate(version_level, version_reserved);
    fillReserved(format, masked_data);
}

void QRCodeEncoderImpl::generatingProcess(const std::string& input, Mat& final_result)
{
    vector<vector<uint8_t> > data_blocks, ecc_blocks;
    if (!stringToBits(input))
    {
        return;
    }
    padBitStream();
    eccGenerate(data_blocks, ecc_blocks);
    rearrangeBlocks(data_blocks, ecc_blocks);
    structureFinalMessage();
    final_result = masked_data.clone();
    const int border = 2;
    copyMakeBorder(final_result, final_result, border, border, border, border, BORDER_CONSTANT, Scalar(255));
}

void QRCodeEncoderImpl::encode(const String& input, OutputArray output)
{
    if (output.kind() != _InputArray::MAT)
        CV_Error(Error::StsBadArg, "Output should be cv::Mat");
    CV_Check((int)mode_type, mode_type != MODE_STRUCTURED_APPEND, "For structured append mode please call encodeStructuredAppend() method");
    CV_Check(struct_num, struct_num == 1, "For structured append mode please call encodeStructuredAppend() method");
    generateQR(input);
    CV_Assert(!final_qrcodes.empty());
    output.assign(final_qrcodes[0]);
    final_qrcodes.clear();
}

void QRCodeEncoderImpl::encodeStructuredAppend(const String& input, OutputArrayOfArrays output)
{
    if (output.kind() != _InputArray::STD_VECTOR_MAT)
        CV_Error(Error::StsBadArg, "Output should be vector of cv::Mat");
    mode_type = MODE_STRUCTURED_APPEND;
    generateQR(input);
    CV_Assert(!final_qrcodes.empty());
    output.create((int)final_qrcodes.size(), 1, final_qrcodes[0].type());
    vector<Mat> dst;
    output.getMatVector(dst);
    for (int i = 0; i < (int)final_qrcodes.size(); i++)
    {
        output.getMatRef(i) = final_qrcodes[i];
    }
    final_qrcodes.clear();
}

Ptr<QRCodeEncoder> QRCodeEncoder::create(const QRCodeEncoder::Params& parameters)
{
    return makePtr<QRCodeEncoderImpl>(parameters);
}

class QRCodeDecoderImpl : public QRCodeDecoder {
public:
    bool decode(const Mat& straight, String& decoded_info) CV_OVERRIDE;

private:
    QRCodeEncoder::CorrectionLevel level;
    int version;

    struct Bitstream {
        int next(int bits) {
            CV_Assert(idx < data.size());

            int val = 0;
            while (bits >= actualBits) {
                CV_CheckLT(idx, data.size(), "Not enough bits in the bitstream");
                val |= data[idx++] << (bits - actualBits);
                bits -= actualBits;
                actualBits = 8;
            }
            if (bits && idx < data.size()) {
                val |= data[idx] >> (actualBits - bits);
                actualBits -= bits;
                data[idx] &= 255 >> (8 - actualBits);
            }
            return val;
        }

        bool empty() {
            return idx >= data.size();
        }

        std::vector<uint8_t> data;
        int actualBits = 8;
        size_t idx = 0;
    } bitstream;

    bool run(const Mat& straight, String& decoded_info);
    bool decodeFormatInfo(const Mat& straight, int& mask);
    bool correctFormatInfo(uint16_t& format_info);
    void extractCodewords(Mat& source, std::vector<uint8_t>& codewords);
    bool errorCorrection(std::vector<uint8_t>& codewords);
    bool errorCorrectionBlock(std::vector<uint8_t>& codewords);
    void decodeSymbols(String& result);
    void decodeNumeric(String& result);
    void decodeAlpha(String& result);
    void decodeByte(String& result);
    void decodeECI(String& result);
    void decodeKanji(String& result);
    void decodeStructuredAppend(String& result);
};

QRCodeDecoder::~QRCodeDecoder()
{
    // nothing
}

Ptr<QRCodeDecoder> QRCodeDecoder::create() {
    return makePtr<QRCodeDecoderImpl>();
}

bool QRCodeDecoderImpl::decode(const Mat& _straight, String& decoded_info) {
    Mat straight = ~_straight;  // Invert modules
    bool decoded = run(straight, decoded_info);
    if (!decoded) {
        cv::transpose(straight, straight);
        decoded = run(straight, decoded_info);
    }
    return decoded;
}

// Unmask format info bits and apply error correction
bool QRCodeDecoderImpl::correctFormatInfo(uint16_t& format_info) {
    static const uint16_t mask_pattern = 0b101010000010010;

    cv::Hamming hd;
    for (int i = 0; i < 32; ++i) {
        // Compute Hamming distance
        int distance = hd(reinterpret_cast<const unsigned char*>(&formatInfoLUT[i]),
                          reinterpret_cast<const unsigned char*>(&format_info), 2);
        // Up to 3 bit errors might be corrected.
        // So if distance is less or equal than 3 - we found a correct format info.
        if (distance <= 3) {
            format_info = formatInfoLUT[i] ^ mask_pattern;
            return true;
        }
    }
    return false;
}

bool QRCodeDecoderImpl::decodeFormatInfo(const Mat& straight, int& mask) {
    // Read left-top format info
    uint16_t format_info = 0;
    for (int i = 0; i < 6; ++i)
        format_info |= (straight.at<uint8_t>(i, 8) & 1) << i;

    format_info |= (straight.at<uint8_t>(7, 8) & 1) << 6;
    format_info |= (straight.at<uint8_t>(8, 8) & 1) << 7;
    format_info |= (straight.at<uint8_t>(8, 7) & 1) << 8;

    for (int i = 9; i < 15; ++i)
        format_info |= (straight.at<uint8_t>(8, 14 - i) & 1) << i;

    bool correct = correctFormatInfo(format_info);

    // Format information 15bit sequence appears twice.
    // Try extract format info from different position.
    uint16_t format_info_dup = 0;
    for (int i = 0; i < 8; ++i)
        format_info_dup |= (straight.at<uint8_t>(8, straight.cols - 1 - i) & 1) << i;
    for (int i = 0; i < 7; ++i)
        format_info_dup |= (straight.at<uint8_t>(straight.rows - 7 + i, 8) & 1) << (i + 8);

    if (correctFormatInfo(format_info_dup)) {
        // Both strings must be the same
        if (correct && format_info != format_info_dup)
            return false;
        format_info = format_info_dup;
    } else {
        if (!correct)
            return false;
    }

    switch((format_info >> 13) & 0b11) {
        case 0: level = QRCodeEncoder::CorrectionLevel::CORRECT_LEVEL_M; break;
        case 1: level = QRCodeEncoder::CorrectionLevel::CORRECT_LEVEL_L; break;
        case 2: level = QRCodeEncoder::CorrectionLevel::CORRECT_LEVEL_H; break;
        case 3: level = QRCodeEncoder::CorrectionLevel::CORRECT_LEVEL_Q; break;
    };
    mask = (format_info >> 10) & 0b111;
    return true;
}

bool QRCodeDecoderImpl::run(const Mat& straight, String& decoded_info) {
    CV_Assert(straight.rows == straight.cols);
    version = (straight.rows - 21) / 4 + 1;

    decoded_info = "";
    mode = static_cast<QRCodeEncoder::EncodeMode>(0);
    eci = static_cast<QRCodeEncoder::ECIEncodings>(0);

    // Decode format info
    int maskPattern;
    bool decoded = decodeFormatInfo(straight, maskPattern);
    if (!decoded) {
        return false;
    }

    // Generate data mask
    Mat masked = straight.clone();
    maskData(straight, maskPattern, masked);

    extractCodewords(masked, bitstream.data);
    if (!errorCorrection(bitstream.data)) {
        return false;
    }
    decodeSymbols(decoded_info);
    return true;
}

bool QRCodeDecoderImpl::errorCorrection(std::vector<uint8_t>& codewords) {
    CV_CheckEQ((int)codewords.size(), version_info_database[version].total_codewords,
               "Number of codewords");

    int numBlocks = version_info_database[version].ecc[level].num_blocks_in_G1 +
                    version_info_database[version].ecc[level].num_blocks_in_G2;
    if (numBlocks == 1) {
        return errorCorrectionBlock(codewords);
    }

    size_t numData = 0;
    std::vector<int> blockSizes;
    blockSizes.reserve(numBlocks);
    for (int i = 0; i < version_info_database[version].ecc[level].num_blocks_in_G1; ++i) {
        blockSizes.push_back(version_info_database[version].ecc[level].data_codewords_in_G1);
        numData += blockSizes.back();
    }
    for (int i = 0; i < version_info_database[version].ecc[level].num_blocks_in_G2; ++i) {
        blockSizes.push_back(version_info_database[version].ecc[level].data_codewords_in_G2);
        numData += blockSizes.back();
    }

    // TODO: parallel_for
    std::vector<std::vector<uint8_t>> blocks(numBlocks);
    int minBlockSize = *std::min_element(blockSizes.begin(), blockSizes.end());
    size_t offset = 0;
    for (int i = 0; i < minBlockSize; ++i) {
        for (int j = 0; j < numBlocks; ++j) {
            blocks[j].push_back(codewords[offset++]);
        }
    }
    // Put remaining data codewords
    for (int j = 0; j < numBlocks; ++j) {
        CV_Assert(blockSizes[j] == minBlockSize || blockSizes[j] == minBlockSize + 1);
        if (blockSizes[j] > minBlockSize)
            blocks[j].push_back(codewords[offset++]);
    }
    // Copy error correction codewords
    int numEcc = version_info_database[version].ecc[level].ecc_codewords;
    for (int i = 0; i < numEcc; ++i) {
        for (int j = 0; j < numBlocks; ++j) {
            blocks[j].push_back(codewords[offset++]);
        }
    }

    parallel_for_(Range(0, numBlocks), [&](const Range& r) {
        for (int i = r.start; i < r.end; ++i) {
            if (!errorCorrectionBlock(blocks[i])) {
                blocks[i].clear();
                return;
            }
        }
    });

    // Collect blocks back after error correction. Trim error correction codewords.
    codewords.resize(numData);
    offset = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].empty())
            return false;
        std::copy(blocks[i].begin(), blocks[i].end(), codewords.begin() + offset);
        offset += blocks[i].size();
    }

    return true;
}

bool QRCodeDecoderImpl::errorCorrectionBlock(std::vector<uint8_t>& codewords) {
    size_t numEcc = version_info_database[version].ecc[level].ecc_codewords;
    size_t numSyndromes = numEcc;

    // According to the ISO there is a formula for a number of the syndromes.
    // However several tests don't pass the error correction step because of less number of syndromes:
    // 1M: qrcodes/detection/lots/image001.jpg from BoofCV (8 syndromes by formula, 10 needed)
    // 1L: Objdetect_QRCode_Multi.regression/13 (4 syndromes by formula, 6 needed)
    // 2L: qrcodes/detection/brightness/image011.jpg from BoofCV (8 syndromes by formula, 10 needed)
    if (numSyndromes % 2 == 1)
        numSyndromes -= 1;

    // Compute syndromes
    bool hasError = false;
    std::vector<uint8_t> syndromes(numSyndromes, codewords[0]);
    for (size_t i = 0; i < syndromes.size(); ++i) {
        for (size_t j = 1; j < codewords.size(); ++j) {
            syndromes[i] = gfMul(syndromes[i], gfPow(2, static_cast<int>(i))) ^ codewords[j];
        }
        hasError |= syndromes[i] != 0;
    }
    if (!hasError) {
        // Trim error correction codewords
        codewords.resize(codewords.size() - numEcc);
        return true;
    }

    // Run Berlekamp–Massey algorithm to find error positions (coefficients of locator poly)
    size_t L = 0;   // number of assumed errors
    size_t m = 1;   // shift value (between C and B)
    uint8_t b = 1;  // discrepancy from last L update

    std::vector<uint8_t> C(numSyndromes, 0);  // Error locator polynomial
    std::vector<uint8_t> B(numSyndromes, 0);  // A copy of error locator from previos L update
    C[0] = B[0] = 1;
    for (size_t i = 0; i < numSyndromes; ++i) {
        CV_Assert(m + L - 1 < C.size());  // m >= 1 on any iteration
        uint8_t discrepancy = syndromes[i];
        for (size_t j = 1; j <= L; ++j) {
            discrepancy ^= gfMul(C[j], syndromes[i - j]);
        }

        if (discrepancy == 0) {
            m += 1;
        } else {
            std::vector<uint8_t> C_copy = C;
            uint8_t inv_b = gfDiv(1, b);
            uint8_t tmp = gfMul(discrepancy, inv_b);

            for (size_t j = 0; j < L; ++j) {
                C[m + j] ^= gfMul(tmp, B[j]);
            }

            if (2 * L <= i) {
                L = i + 1 - L;
                B = C_copy;
                b = discrepancy;
                m = 1;
            } else {
                m += 1;
            }
        }
    }

    // There is an error at i-th position if i is a root of locator poly
    std::vector<size_t> errLocs;
    errLocs.reserve(L);
    for (size_t i = 0; i < codewords.size(); ++i) {
        uint8_t val = 1;
        uint8_t pos = gfPow(2, static_cast<int>(i));
        for (size_t j = 1; j <= L; ++j) {
            val = gfMul(val, pos) ^ C[j];
        }
        if (val == 0) {
            errLocs.push_back(static_cast<int>(codewords.size() - 1 - i));
        }
    }

    // Number of assumed errors does not match number of error locations
    if (errLocs.size() != L)
        return false;

    // Forney algorithm for error correction using syndromes and known error locations
    std::vector<uint8_t> errEval;
    gfPolyMul(C, syndromes, errEval);

    for (size_t i = 0; i < errLocs.size(); ++i) {
        uint8_t numenator = 0, denominator = 0;
        uint8_t X = gfPow(2, static_cast<int>(codewords.size() - 1 - errLocs[i]));
        uint8_t inv_X = gfDiv(1, X);

        for (size_t j = 0; j < L; ++j) {
            numenator = gfMul(numenator, inv_X) ^ errEval[L - 1 - j];
        }

        // Compute demoninator as a product of (1-X_i * X_k) for i != k
        // TODO: optimize, there is a dubplicated compute
        denominator = 1;
        for (size_t j = 0; j < errLocs.size(); ++j) {
            if (i == j)
                continue;
            uint8_t Xj = gfPow(2, static_cast<int>(codewords.size() - 1 - errLocs[j]));
            denominator = gfMul(denominator, 1 ^ gfMul(inv_X, Xj));
        }

        uint8_t errValue = gfDiv(numenator, denominator);
        codewords[errLocs[i]] ^= errValue;
    }

    // Trim error correction codewords
    codewords.resize(codewords.size() - numEcc);
    return true;
}

void QRCodeDecoderImpl::extractCodewords(Mat& source, std::vector<uint8_t>& codewords) {
    const VersionInfo& version_info = version_info_database[version];

    // Mask alignment markers
    std::vector<int> alignCenters;
    alignCenters.reserve(MAX_ALIGNMENT);
    for (int i = 0; i < MAX_ALIGNMENT && version_info.alignment_pattern[i]; i++)
        alignCenters.push_back(version_info.alignment_pattern[i]);

    for (size_t i = 0; i < alignCenters.size(); i++)
    {
        for (size_t j = 0; j < alignCenters.size(); j++)
        {
            if ((i == alignCenters.size() - 1 && j == 0) || (i == 0 && j == 0) ||
                (j == alignCenters.size() - 1 && i == 0))
                continue;
            int x = alignCenters[i];
            int y = alignCenters[j];
            Mat area = source({x - 2, x + 3}, {y - 2, y + 3});
            area.setTo(INVALID_REGION_VALUE);
        }
    }

    // Mask detection markers
    source.rowRange(0, 9).colRange(source.cols - 8, source.cols).setTo(INVALID_REGION_VALUE);
    source.rowRange(0, 9).colRange(0, 9).setTo(INVALID_REGION_VALUE);
    source.colRange(0, 9).rowRange(source.rows - 8, source.rows).setTo(INVALID_REGION_VALUE);

    // Mask Version Information blocks
    if (version >= 7) {
        source.rowRange(0, 6).colRange(source.cols - 12, source.cols - 9).setTo(INVALID_REGION_VALUE);
        source.colRange(0, 6).rowRange(source.rows - 12, source.rows - 9).setTo(INVALID_REGION_VALUE);
    }

    // Mask timing pattern
    source.row(6) = INVALID_REGION_VALUE;

    std::vector<uint8_t> bits;
    bits.reserve(source.total() - source.cols);
    bool moveUpwards = true;
    for (auto& data : {source.colRange(7, source.cols), source.colRange(0, 6)}) {
        for (int i = data.cols / 2 - 1; i >= 0; --i) {
            Mat col0 = data.col(i * 2);
            Mat col1 = data.col(i * 2 + 1);
            for (int j = 0; j < data.rows; ++j) {
                if (moveUpwards) {
                    bits.push_back(col1.at<uint8_t>(data.rows - 1 - j));
                    bits.push_back(col0.at<uint8_t>(data.rows - 1 - j));
                } else {
                    bits.push_back(col1.at<uint8_t>(j));
                    bits.push_back(col0.at<uint8_t>(j));
                }
            }
            moveUpwards = !moveUpwards;
        }
    }

    // Combine bits to codewords
    size_t numCodewords = version_info.total_codewords;
    codewords.resize(numCodewords);

    size_t offset = 0;
    for (size_t i = 0; i < numCodewords; ++i) {
        codewords[i] = 0;
        for (size_t j = 0; j < 8; ++j) {
            while (bits[offset] == INVALID_REGION_VALUE) {
                offset += 1;
                CV_Assert(offset < bits.size());
            }
            codewords[i] |= (bits[offset] & 1) << (7 - j);
            offset += 1;
        }
    }
}

void QRCodeDecoderImpl::decodeSymbols(String& result) {
    CV_Assert(!bitstream.empty());

    // Decode depends on the mode
    result = "";
    while (!bitstream.empty()) {
        // Determine mode
        auto currMode = static_cast<QRCodeEncoder::EncodeMode>(bitstream.next(4));
        if (this->mode == 0) {
            mode = currMode;
        }

        if (currMode == 0 || bitstream.empty())
            return;
        if (currMode == QRCodeEncoder::EncodeMode::MODE_NUMERIC)
            decodeNumeric(result);
        else if (currMode == QRCodeEncoder::EncodeMode::MODE_ALPHANUMERIC)
            decodeAlpha(result);
        else if (currMode == QRCodeEncoder::EncodeMode::MODE_BYTE)
            decodeByte(result);
        else if (currMode == QRCodeEncoder::EncodeMode::MODE_ECI)
            decodeECI(result);
        else if (currMode == QRCodeEncoder::EncodeMode::MODE_KANJI)
            decodeKanji(result);
        else if (currMode == QRCodeEncoder::EncodeMode::MODE_STRUCTURED_APPEND) {
            sequence_num = static_cast<uint8_t>(bitstream.next(4));
            total_num = static_cast<uint8_t>(1 + bitstream.next(4));
            parity = static_cast<uint8_t>(bitstream.next(8));
        }
        else
            CV_Error(Error::StsNotImplemented, format("mode %d", currMode));
    }
}

void QRCodeDecoderImpl::decodeNumeric(String& result) {
    int numDigits = bitstream.next(version <= 9 ? 10 : (version <= 26 ? 12 : 14));
    for (int i = 0; i < numDigits / 3; ++i) {
        int triple = bitstream.next(10);
        result += static_cast<char>('0' + triple / 100);
        result += static_cast<char>('0' + (triple / 10) % 10);
        result += static_cast<char>('0' + triple % 10);
    }
    int remainingDigits = numDigits % 3;
    if (remainingDigits) {
        int triple = bitstream.next(remainingDigits == 1 ? 4 : 7);
        if (remainingDigits == 2)
            result += '0' + (triple / 10) % 10;
        result += '0' + triple % 10;
    }
}

void QRCodeDecoderImpl::decodeAlpha(String& result) {
    static const char map[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                               'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '$', '%', '*',
                               '+', '-', '.', '/', ':'};

    int num = bitstream.next(version <= 9 ? 9 : (version <= 26 ? 11 : 13));
    for (int i = 0; i < num / 2; ++i) {
        int tuple = bitstream.next(11);
        result += map[tuple / 45];
        result += map[tuple % 45];
    }
    if (num % 2) {
        int value = bitstream.next(6);
        result += map[value];
    }
}

void QRCodeDecoderImpl::decodeByte(String& result) {
    int num = bitstream.next(version <= 9 ? 8 : 16);
    for (int i = 0; i < num; ++i) {
        result += static_cast<char>(bitstream.next(8));
    }
}

void QRCodeDecoderImpl::decodeECI(String& result) {
    int eciAssignValue = bitstream.next(8);
    for (int i = 0; i < 8; ++i) {
        if (eciAssignValue & 1 << (7 - i))
            eciAssignValue |= bitstream.next(8) << (i + 1) * 8;
        else
            break;
    }
    if (this->eci == 0) {
        this->eci = static_cast<QRCodeEncoder::ECIEncodings>(eciAssignValue);
    }
    decodeSymbols(result);

}

void QRCodeDecoderImpl::decodeKanji(String& result) {
    int num = bitstream.next(version <= 9 ? 8 : (version <= 26 ? 10 : 12));
    for (int i = 0; i < num; ++i) {
        int data = bitstream.next(13);
        int high_byte = data / 0xC0;
        int low_byte = data - high_byte * 0xC0;
        int symbol = (high_byte << 8) + low_byte;
        if (0 <= symbol && symbol <= 0x9FFC - 0x8140) {
            symbol += 0x8140;
        } else if (0xE040 - 0xC140 <= symbol && symbol <= 0xEBBF - 0xC140) {
            symbol += 0xC140;
        }
        result += (symbol >> 8) & 0xff;
        result += symbol & 0xff;
    }
}

}

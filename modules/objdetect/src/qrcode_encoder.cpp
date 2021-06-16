// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Intel Corporation, all rights reserved.

#include "precomp.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

namespace cv
{
using std::vector;

const int max_payload_len = 8896;
const int max_format_length = 15;
const int max_version_length = 18;
const int max_version = 40;
const int max_alignment = 7;
const int error_mode_occur = 99999;

const uint8_t invalid_region_value = 110;

std::string decToBin(const int &format, const int &total);
int eccLevelToCode(int level);
uint8_t gfPow(uint8_t x, int power);
uint8_t gfMul(const uint8_t& x, const uint8_t& y);
Mat gfPolyMul(const Mat& p, const Mat& q);
Mat gfPolyDiv(const Mat& dividend, const Mat& divisor, const int& ecc_num);
Mat polyGenerator(const int& n);
int getBits(const int& bits, const vector<uint8_t>& payload ,int& pay_index);
void loadString(const std::string& str, vector<uint8_t>& cur_str, bool is_bit_stream);

std::string decToBin(const int& format, const int& total)
{
    std::string f;
    int num = total;
    for(int i = format; num > 0; i = i >> 1, num--)
    {
        if(i % 2 == 1)
            f = '1' + f;
        else
            f = '0' + f;
    }
    return f;
}

void loadString(const std::string& str, vector<uint8_t>& cur_str, bool is_bit_stream = false)
{
    for(size_t i = 0; i < str.length(); i++)
    {
        if(is_bit_stream)
        {
            cur_str.push_back(uint8_t(str[i] - '0'));
        }
        else
        {
            cur_str.push_back(uint8_t(str[i]));
        }
    }
    return;
}

struct BlockParams
{
    int ecc_codewords;
    int num_blocks_in_G1;
    int data_codewords_in_G1;
    int num_blocks_in_G2;
    int data_codewords_in_G2;
};

struct VersionInfo
{
    int	total_codewords;
    int	alignment_pattern[max_alignment];
    BlockParams ecc[4];
};

/**int numeric_mode;
int alpha_mode;
int byte_mode;
int kanji_mode;*/
struct ECLevelCapacity
{
    int encoding_modes[4];
};
struct CharacterCapacity
{
    ECLevelCapacity ec_level[4];
};
const CharacterCapacity version_capacity_database[max_version + 1] =
{
        {
            {
                {0, 1, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}
            }
        },
        {
            {
                {41, 25, 17, 10},
                {34, 20, 14, 8},
                {27, 16, 11, 7},
                {17, 10, 7 , 4}
            }
        },
        {
            {
                {77, 47, 32, 20},
                {63, 38, 26, 16},
                {48, 29, 20, 12},
                {34, 20, 14, 8}
            }
        },
        {
            {
                {127, 77, 53, 32},
                {101, 61, 42, 26},
                {77, 47, 32, 20},
                {58, 35, 24, 15}
            }
        },
        {
            {
                {187, 114, 78, 48},
                {149, 90, 62, 38},
                {111, 67, 46, 28},
                {82, 50, 34, 21}
            }
        },
        {
            {
                {255, 154, 106, 65},
                {202, 122, 84, 52},
                {144, 87, 60, 37},
                {106, 64, 44, 27}
            }
        },
        {
            {
                {322, 195, 134, 82},
                {255, 154, 106, 65},
                {178, 108, 74, 45},
                {139, 84 , 58, 36}
            }
        },
        {
            {
                {370, 224, 154, 95},
                {293, 178, 122, 75},
                {207, 125, 86, 53},
                {154, 93 , 64, 39}
            }
        },
        {
            {
                {461, 279, 192, 118},
                {365, 221, 152, 93},
                {259, 157, 108, 66},
                {202, 122, 84, 52}
            }
        },
        {
            {
                {552, 335, 230, 141},
                {432, 262, 180, 111},
                {312, 189, 130, 80},
                {235, 143, 98, 60}
            }
        },
        {
            {
                {652, 395, 271, 167},
                {513, 311, 213, 131},
                {364, 221, 151, 93},
                {288, 174, 119, 74}
            }
        },
        {
            {
                {772, 468, 321, 198},
                {604, 366, 251, 155},
                {427, 259, 177, 109},
                {331, 200, 137, 85}
            }
        },
        {
            {
                {883, 535, 367, 226},
                {691, 419, 287, 177},
                {489, 296, 203, 125},
                {374, 227, 155, 96}
            }
        },
        {
            {
                {1022, 619, 425, 262},
                {796, 483, 331,	204},
                {580, 352, 241,	149},
                {427, 259, 177,	109}
            }
        },
        {
            {
                {1101,	667, 458, 282},
                {871, 528, 362,	223},
                {621, 376, 258,	159},
                {468, 283, 194,	120}
            }
        },
        {
            {
                {1250, 758, 520, 320},
                {991, 600, 412, 254},
                {703, 426, 292, 180},
                {530, 321, 220, 136}
            }
        },
        {
            {
                {1408, 854, 586, 361},
                {1082, 656, 450, 277},
                {775, 470, 322, 198},
                {602, 365, 250, 154}
            }
        },
        {
            {
                {1548, 938, 644, 397},
                {1212, 734, 504, 310},
                {876, 531, 364, 224},
                {674, 408, 280,	173}
            }
        },
        {
            {
                {1725, 1046, 718, 442},
                {1346, 816, 560, 345},
                {948, 574, 394, 243},
                {746, 452, 310, 191}
            }
        },
        {
            {
                {1903, 1153, 792, 488},
                {1500, 909,	624, 384},
                {1063, 644,	442, 272},
                {813, 493, 338,	208}
            }
        },
        {
            {
                {2061, 1249, 858, 528},
                {1600, 970, 666, 410},
                {1159, 702, 482, 297},
                {919, 557, 382, 235}
            }
        },
        {
            {
                {2232,	1352, 929, 572},
                {1708, 1035, 711, 438},
                {1224, 742,	509, 314},
                {969, 587, 403,	248}
            }
        },
        {
            {
                {2409, 1460, 1003, 618},
                {1872, 1134, 779, 480},
                {1358, 823, 565, 348},
                {1056, 640, 439, 270}
            }
        },
        {
            {
                {2620,	1588, 1091, 672},
                {2059, 1248, 857, 528},
                {1468, 890, 611, 376},
                {1108, 672, 461, 284}
            }
        },
        {
            {
                {2812, 1704, 1171, 721},
                {2188, 1326, 911, 561},
                {1588, 963,	661, 407},
                {1228, 744,	511, 315}
            }
        },
        {
            {
                {3057,	1853, 1273, 784},
                {2395, 1451, 997, 614},
                {1718, 1041, 715, 440},
                {1286, 779, 535, 330}
            }
        },
        {
            {
                {3283, 1990, 1367, 842},
                {2544, 1542, 1059, 652},
                {1804, 1094, 751, 462},
                {1425, 864, 593, 365}
            }
        },
        {
            {
                {3517,	2132, 1465, 902},
                {2701, 1637, 1125, 692},
                {1933, 1172, 805, 496},
                {1501, 910, 625, 385}
            }
        },
        {
            {
                {3669,	2223, 1528, 940},
                {2857, 1732, 1190, 732},
                {2085, 1263, 868, 534},
                {1581, 958, 658, 405}
            }
        },
        {
            {
                {3909,	2369, 1628, 1002},
                {3035, 1839, 1264, 778},
                {2181, 1322, 908, 559},
                {1677, 1016, 698, 430}
            }
        },
        {
            {
                {4158, 2520, 1732, 1066},
                {3289, 1994, 1370, 843},
                {2358, 1429, 982, 604},
                {1782, 1080, 742, 457}
            }
        },
        {
            {
                {4417, 2677, 1840, 1132},
                {3486, 2113, 1452, 894},
                {2473, 1499, 1030, 634},
                {1897, 1150, 790, 486}
            }

        },
        {
            {
                {4686,	2840, 1952, 1201},
                {3693, 2238, 1538, 947},
                {2670, 1618, 1112, 684},
                {2022, 1226, 842, 518}
            }
        },
        {
            {
                {4965, 3009, 2068, 1273},
                {3909, 2369, 1628, 1002},
                {2805, 1700, 1168, 719},
                {2157, 1307, 898, 553}
            }
        },
        {
            {
                {5253, 3183, 2188, 1347},
                {4134, 2506, 1722, 1060},
                {2949, 1787, 1228, 756},
                {2301, 1394, 958 , 590}
            }
        },
        {
            {
                {5529,	3351, 2303, 1417},
                {4343, 2632, 1809, 1113},
                {3081, 1867, 1283, 790},
                {2361, 1431, 983, 605}
            }
        },
        {
            {
                {5836,	3537, 2431, 1496},
                {4588, 2780, 1911, 1176},
                {3244, 1966, 1351, 832},
                {2524, 1530, 1051, 647}
            }
        },
        {
            {
                {6153, 3729, 2563, 1577},
                {4775, 2894, 1989, 1224},
                {3417, 2071, 1423, 876},
                {2625, 1591, 1093, 673}
            }
        },
        {
            {
                {6479, 3927, 2699, 1661},
                {5039, 3054, 2099, 1292},
                {3599, 2181, 1499, 923},
                {2735, 1658, 1139, 701}
            }
        },
        {
            {
                {6743, 4087, 2809, 1729},
                {5313, 3220, 2213, 1362},
                {3791, 2298, 1579, 972},
                {2927, 1774, 1219, 750}
            }
        },
        {
            {
                {7089, 4296, 2953, 1817},
                {5596, 3391, 2331, 1435},
                {3993, 2420, 1663, 1024},
                {3057, 1852, 1273, 784}
            }
        }
};

const VersionInfo version_info_database[max_version + 1] =
{
        { /* Version 0 */
            0,
            {0, 0, 0, 0, 0, 0, 0},
            {
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0}
            }
        },
        { /* Version 1 */
            26,
            {0, 0, 0, 0, 0, 0, 0},
            {
                {7, 1, 19, 0, 0},
                {10, 1, 16, 0, 0},
                {13, 1, 13, 0, 0},
                {17, 1, 9 , 0, 0}
            }
        },
        { /* Version 2 */
            44,
            {6, 18, 0, 0, 0, 0, 0},
            {
                {10, 1, 34, 0, 0},
                {16, 1, 28, 0, 0},
                {22, 1,	22, 0, 0},
                {28, 1,	16, 0, 0}
            }
        },
        { /* Version 3 */
            70,
            {6, 22, 0, 0, 0, 0, 0},
            {
                {15, 1,	55, 0, 0},
                {26, 1,	44, 0, 0},
                {18, 2,	17, 0, 0},
                {22, 2, 13, 0, 0}
            }
        },
        { /* Version 4 */
            100,
            {6, 26, 0,0,0,0,0},
            {
                {20, 1,	80, 0, 0},
                {18, 2,	32, 0, 0},
                {26, 2,	24, 0, 0},
                {16, 4,	9 , 0, 0}
            }
        },
        { /* Version 5 */
            134,
            {6, 30, 0, 0, 0, 0, 0},
            {
                {26, 1, 108, 0, 0},
                {24, 2, 43, 0, 0},
                {18, 2, 15, 2, 16},
                {22, 2, 11, 2, 12}
            }
        },
        { /* Version 6 */
            172,
            {6, 34, 0, 0, 0, 0, 0},
            {
                {18, 2, 68, 0, 0},
                {16, 4, 27, 0, 0},
                {24, 4, 19, 0, 0},
                {28, 4, 15, 0, 0}
            }
        },
        { /* Version 7 */
            196,
            {6, 22, 38, 0, 0, 0, 0},
            {
                {20, 2, 78, 0, 0},
                {18, 4, 31, 0, 0},
                {18, 2,	14, 4, 15},
                {26, 4,	13, 1, 14}
            }
        },
        { /* Version 8 */
            242,
            {6, 24, 42, 0, 0, 0, 0},
            {
                {24, 2,	97, 0, 0},
                {22, 2,	38, 2, 39},
                {22, 4,	18, 2, 19},
                {26, 4,	14, 2, 15}
            }
        },
        { /* Version 9 */
            292,
            {6, 26, 46, 0, 0, 0, 0},
            {
                {30, 2,	116, 0, 0},
                {22, 3,	36, 2, 37},
                {20, 4,	16, 4, 17},
                {24, 4,	12, 4, 13}
            }
        },
        { /* Version 10 */
            346,
            {6, 28, 50, 0, 0, 0, 0},
            {
                {18, 2,	68, 2, 69},
                {26, 4,	43, 1, 44},
                {24, 6, 19, 2, 20},
                {28, 6, 15, 2, 16}
            }
        },
        { /* Version 11 */
            404,
            {6, 30, 54, 0, 0, 0, 0},
            {
                {20, 4,	81, 0, 0},
                {30, 1,	50, 4, 51},
                {28, 4,	22, 4, 23},
                {24, 3,	12, 8, 13}
            }
        },
        { /* Version 12 */
            466,
            {6, 32, 58, 0, 0, 0, 0},
            {
                {24, 2, 92, 2, 93},
                {22, 6, 36, 2, 37},
                {26, 4,	20, 6, 21},
                {28, 7,	14, 4, 15}
            }
        },
        { /* Version 13 */
            532,
            {6, 34, 62, 0, 0, 0, 0},
            {
                {26, 4, 107, 0, 0},
                {22, 8, 37, 1, 38},
                {24, 8, 20, 4, 21},
                {22, 12, 11, 4, 12}
            }
        },
        { /* Version 14 */
            581,
            {6, 26, 46, 66, 0, 0, 0},
            {
                {30, 3,	115, 1, 116},
                {24, 4,	40, 5, 41},
                {20, 11, 16, 5, 17},
                {24, 11, 12, 5, 13}
            }
        },
        { /* Version 15 */
            655,
            {6, 26, 48, 70, 0, 0, 0},
            {
                {22, 5, 87, 1, 88},
                {24, 5, 41, 5, 42},
                {30, 5,	24, 7, 25},
                {24, 11, 12, 7, 13}
            }
        },
        { /* Version 16 */
            733,
            {6, 26, 50, 74, 0, 0, 0},
            {
                {24, 5,	98, 1, 99},
                {28, 7,	45, 3, 46},
                {24, 15, 19, 2, 20},
                {30, 3,	15, 13, 16}
            }
        },
        { /* Version 17 */
            815,
            {6, 30, 54, 78, 0, 0, 0},
            {
                {28, 1, 107, 5,	108},
                {28, 10, 46, 1,	47},
                {28, 1, 22, 15,	23},
                {28, 2, 14, 17, 15}
            }
        },
        { /* Version 18 */
            901,
            {6, 30, 56, 82, 0, 0, 0},
            {
                {30, 5, 120, 1, 121},
                {26, 9, 43, 4, 44},
                {28, 17, 22, 1, 23},
                {28, 2, 14, 19, 15}
            }
        },
        { /* Version 19 */
            991,
            {6, 30, 58, 86, 0, 0, 0},
            {
                {28, 3, 113, 4,	114},
                {26, 3, 44, 11,	45},
                {26, 17, 21, 4,	22},
                {26, 9, 13, 16,	14}
            }
        },
        { /* Version 20 */
            1085,
            {6, 34, 62, 90, 0, 0, 0},
            {
                {28, 3,	107, 5, 108},
                {26, 3,	41, 13, 42},
                {30, 15, 24, 5,	25},
                {28, 15, 15, 10, 16}
            }
        },
        { /* Version 21 */
            1156,
            {6, 28, 50, 72, 92, 0, 0},
            {
                {28, 4, 116, 4, 117},
                {26, 17, 42, 0, 0},
                {28, 17, 22, 6,	23},
                {30, 19, 16, 6,	17}
            }
        },
        { /* Version 22 */
            1258,
            {6, 26, 50, 74, 98, 0, 0},
            {
                {28, 2, 111, 7, 112},
                {28, 17, 46, 0, 0},
                {30, 7,	24, 16, 25},
                {24, 34, 13, 0, 0}
            }
        },
        { /* Version 23 */
            1364,
            {6, 30, 54, 78, 102, 0, 0},
            {
                {30, 4,	121,5, 122},
                {28, 4,	47, 14,	48},
                {30, 11, 24, 14, 25},
                {30, 16, 15, 14, 16}
            }
        },
        { /* Version 24 */
            1474,
            {6, 28, 54, 80, 106, 0, 0},
            {
                {30, 6, 117,4, 118},
                {28, 6,	45, 14,	46},
                {30, 11, 24, 16, 25},
                {30, 30, 16, 2,	17}
            }
        },
        { /* Version 25 */
            1588,
            {6, 32, 58, 84, 110, 0, 0},
            {
                {26, 8, 106, 4, 107},
                {28, 8,	47, 13,	48},
                {30, 7,	24, 22,	25},
                {30, 22, 15, 13, 16}
            }
        },
        { /* Version 26 */
            1706,
            {6, 30, 58, 86, 114, 0, 0},
            {
                {28, 10, 114, 2, 115},
                {28, 19, 46, 4,	47},
                {28, 28, 22, 6,	23},
                {30, 33, 16, 4,	17}
            }
        },
        { /* Version 27 */
            1828,
            {6, 34, 62, 90, 118, 0, 0},
            {
                {30, 8,	122, 4, 123},
                {28, 22, 45, 3,	46},
                {30, 8,	23, 26,	24},
                {30, 12, 15, 28, 16}
            }
        },
        { /* Version 28 */
            1921,
            {6, 26, 50, 74, 98, 122, 0},
            {
                {30, 3, 117, 10, 118},
                {28, 3,	45, 23,	46},
                {30, 4,	24, 31,	25},
                {30, 11, 15, 31, 16}
            }
        },
        { /* Version 29 */
            2051,
            {6, 30, 54, 78, 102, 126, 0},
            {
                {30, 7, 116, 7, 117},
                {28, 21, 45, 7,	46},
                {30, 1,	23, 37,	24},
                {30, 19, 15, 26, 16}
            }
        },
        { /* Version 30 */
            2185,
            {6, 26, 52, 78, 104, 130, 0},
            {
                {30, 5,	115, 10, 116},
                {28, 19, 47, 10, 48},
                {30, 15, 24, 25, 25},
                {30, 23, 15, 25, 16}
            }
        },
        { /* Version 31 */
            2323,
            {6, 30, 56, 82, 108, 134, 0},
            {
                {30, 13, 115, 3, 116},
                {28, 2, 46, 29,	47},
                {30, 42, 24, 1,	25},
                {30, 23, 15, 28, 16}
            }
        },
        { /* Version 32 */
            2465,
            {6, 34, 60, 86, 112, 138, 0},
            {
                {30, 17, 115, 0, 0},
                {28, 10, 46, 23, 47},
                {30, 10, 24, 35, 25},
                {30, 19, 15, 35, 16}
            }
        },
        { /* Version 33 */
            2611,
            {6, 30, 58, 86, 114, 142, 0},
            {
                {30, 17, 115, 1, 116},
                {28, 14, 46, 21, 47},
                {30, 29, 24, 19, 25},
                {30, 11, 15, 46, 16}
            }
        },
        { /* Version 34 */
            2761,
            {6, 34, 62, 90, 118, 146, 0},
            {
                {30, 13, 115, 6, 116},
                {28, 14, 46, 23, 47},
                {30, 44, 24, 7,	25},
                {30, 59, 16, 1,	17}
            }
        },
        { /* Version 35 */
            2876,
            {6, 30, 54, 78, 102, 126, 150},
            {
                {30, 12, 121, 7, 122},
                {28, 12, 47, 26, 48},
                {30, 39, 24, 14, 25},
                {30, 22, 15, 41, 16}
            }
        },
        { /* Version 36 */
            3034,
            {6, 24, 50, 76, 102, 128, 154},
            {
                {30, 6,	121, 14, 122},
                {28, 6,	47, 34,	48},
                {30, 46, 24, 10, 25},
                {30, 2,	15, 64,	16}
            }
        },
        { /* Version 37 */
            3196,
            {6, 28, 54, 80, 106, 132, 158},
            {
                {30, 17, 122, 4, 123},
                {28, 29, 46, 14, 47},
                {30, 49, 24, 10, 25},
                {30, 24, 15, 46, 16}
            }
        },
        { /* Version 38 */
            3362,
            {6, 32, 58, 84, 110, 136, 162},
            {
                {30, 4, 122, 18, 123},
                {28, 13, 46, 32, 47},
                {30, 48, 24, 14, 25},
                {30, 42, 15, 32, 16}
            }
        },
        { /* Version 39 */
            3532,
            {6, 26, 54, 82, 110, 138, 166},
            {
                {30, 20, 117,4, 118},
                {28, 40, 47, 7, 48},
                {30, 43, 24, 22, 25},
                {30, 10, 15, 67, 16}
            }
        },
        { /* Version 40 */
            3706,
            {6, 30, 58, 86, 114, 142, 170},
            {
                {30, 19, 118, 6, 119},
                {28, 18, 47, 31, 48},
                {30, 34, 24, 34, 25},
                {30, 20, 15, 61, 16}
            }
        }
};

static const uint8_t gf_exp[256] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26,
        0x4c, 0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9,
        0x8f, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0,
        0x9d, 0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35,
        0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23,
        0x46, 0x8c, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0,
        0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1,
        0x5f, 0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc,
        0x65, 0xca, 0x89, 0x0f, 0x1e, 0x3c, 0x78, 0xf0,
        0xfd, 0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f,
        0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2,
        0xd9, 0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88,
        0x0d, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce,
        0x81, 0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93,
        0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc,
        0x85, 0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9,
        0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54,
        0xa8, 0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa,
        0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73,
        0xe6, 0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e,
        0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff,
        0xe3, 0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4,
        0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41,
        0x82, 0x19, 0x32, 0x64, 0xc8, 0x8d, 0x07, 0x0e,
        0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6,
        0x51, 0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef,
        0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x09,
        0x12, 0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5,
        0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0x0b, 0x16,
        0x2c, 0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83,
        0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x01
};
static const uint8_t gf_log[256] = {
        0x00, 0xff, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xc6,
        0x03, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b,
        0x04, 0x64, 0xe0, 0x0e, 0x34, 0x8d, 0xef, 0x81,
        0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x08, 0x4c, 0x71,
        0x05, 0x8a, 0x65, 0x2f, 0xe1, 0x24, 0x0f, 0x21,
        0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45,
        0x1d, 0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9,
        0xc9, 0x9a, 0x09, 0x78, 0x4d, 0xe4, 0x72, 0xa6,
        0x06, 0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd,
        0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88,
        0x36, 0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd,
        0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40,
        0x1e, 0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e,
        0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d,
        0xca, 0x5e, 0x9b, 0x9f, 0x0a, 0x15, 0x79, 0x2b,
        0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57,
        0x07, 0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0x0d,
        0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18,
        0xe3, 0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c,
        0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e,
        0x37, 0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd,
        0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61,
        0xf2, 0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e,
        0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2,
        0x1f, 0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76,
        0xc4, 0x17, 0x49, 0xec, 0x7f, 0x0c, 0x6f, 0xf6,
        0x6c, 0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa,
        0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a,
        0xcb, 0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51,
        0x0b, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7,
        0x4f, 0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8,
        0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf
};

int eccLevelToCode(int level)
{
    switch (level)
    {
        case 0:
            return 0b01;
        case 1:
            return 0b00;
        case 2:
            return 0b11;
        case 3:
            return 0b10;
    }
    return -1;
}

struct autoEncodePerBlock
{
    int block_load_len;
    vector<uint8_t>	block_load;
    int encoding_mode;
    autoEncodePerBlock();
};

autoEncodePerBlock::autoEncodePerBlock()
{
    block_load_len = encoding_mode = 0;
    block_load.reserve(max_payload_len);
}

uint8_t gfPow(uint8_t x, int power)
{
    return gf_exp[(gf_log[x] * power) % 255];
}

uint8_t gfMul(const uint8_t&x, const uint8_t& y)
{
    if (x == 0 || y == 0)
        return 0;
    return gf_exp[(gf_log[x] + gf_log[y]) % 255];
}

Mat gfPolyMul(const Mat &p, const Mat &q)
{
    Mat r(1, p.cols+q.cols-1, CV_8UC1, Scalar(0));
    int len_p = p.cols;
    int len_q = q.cols;
    for (int j = 0; j < len_q; j++)
    {
        if (!q.at<uint8_t>(0, j))
            continue;
        for (int i = 0; i < len_p; i++)
        {
            if (!p.at<uint8_t>(0, i))
                continue;
            r.at<uint8_t>(0, i + j) ^= gfMul(p.at<uint8_t>(0, i), q.at<uint8_t>(0, j));
        }
    }
    return r;
}

Mat gfPolyDiv(const Mat& dividend, const Mat& divisor, const int& ecc_num)
{
    int times = dividend.cols - (divisor.cols - 1);
    int dividend_len = dividend.cols - 1;
    int divisor_len = divisor.cols - 1;
    Mat r = dividend.clone();
    for (int i = 0; i < times; i++)
    {
        uint8_t coef = r.at<uint8_t>(0, dividend_len - i);
        if(coef != 0)
        {
            for (int j = 0; j < divisor.cols; j++)
            {
                if (divisor.at<uint8_t>(0, divisor_len - j) != 0)
                {
                    r.at<uint8_t>(0, dividend_len - i - j) ^= gfMul(divisor.at<uint8_t>(0, divisor_len - j), coef);
                }
            }
        }
    }
    Mat ecc = r(Range(0, 1), Range(0, ecc_num)).clone();
    return ecc;
}

Mat polyGenerator(const int& n)
{
    Mat result = (Mat_<uint8_t >(1, 1) << 1);
    Mat temp =   (Mat_<uint8_t >(1, 2) << 1,1);
    for (int i = 1; i <= n; i++)
    {
        temp.at<uint8_t>(0, 0) = gfPow(2, i - 1);
        result = gfPolyMul(result, temp);
    }
    return result;
}

int getBits(const int& bits, const vector<uint8_t>& payload, int& pay_index)
{
    int result = 0;
    for (int i = 0; i < bits; i++)
    {
        result = result << 1;
        result += payload[pay_index++];
    }
    return result;
}

class QREncoder
{
public:
    vector<Mat> final_qrcodes;
    void init(int mode, int version, int ecc, int structure_num);
    void generateQR(const std::string& input);
protected:
    int version_level;
    int ecc_level;
    int mode_type;
    int struct_num;
    int version_size;
    int mask_type;
    Mat format;
    Mat version_reserved;
    std::string input_info;
    vector<uint8_t>	payload;
    vector<uint8_t>	rearranged_data;
    Mat original;
    Mat masked_data;
    uint8_t parity;
    uint8_t sequence_num;
    uint8_t total_num;

    const VersionInfo *version_info ;
    const  BlockParams *cur_ecc_params;
    bool encodeByte(const std::string& input, vector<uint8_t>& output);
    bool encodeAlpha(const std::string& input, vector<uint8_t>& output);
    bool encodeNumeric(const std::string& input, vector<uint8_t>& output);
    bool encodeAuto(const std::string& input, vector<uint8_t>& output);
    bool encodeStructure(const std::string& input, vector<uint8_t>& payload);
    void padBitStream();
    bool stringToBits();
    void eccGenerate(vector<Mat>& data_blocks, vector<Mat>& ecc_blocks);
    void rearrangeBlocks(const vector<Mat>& data_blocks, const vector<Mat>& ecc_blocks);
    void writeReservedArea();
    bool writeBit(int x, int y, bool value);
    void writeData();
    void structureFinalMessage();
    void formatGenerate(const int& mask_type_num, Mat& format_array);
    void versionInfoGenerate(const int& version_level_num, Mat& version_array);
    void fillReserved(const Mat& format_array, Mat& masked);
    void maskData(const int& mask_type_num, Mat& masked);
    void findAutoMaskType();
    QREncodeMode fncModeSelect(const std::string& input);
    bool estimateVersion(const int& input_length, vector<int>& possible_version);
    bool generateBlock(const std::string& input, int mode, autoEncodePerBlock& block);
    int versionAuto(const std::string& input_str);
    int findVersionCapacity(const int &input_length, const int& ecc, const int& version_begin, const int& version_end);
    void generatingProcess(Mat& qrcode);
};

int QREncoder::findVersionCapacity(const int& input_length, const int& ecc, const int& version_begin, const int& version_end)
{
    int data_codewords, version_index = -1;
    const int byte_len = 8;
    version_index = -1;

    for (int i = version_begin; i < version_end; i++)
    {
        const BlockParams * tmp_ecc_params = &(version_info_database[i].ecc[ecc]);
        data_codewords = tmp_ecc_params->data_codewords_in_G1 * tmp_ecc_params->num_blocks_in_G1 +
                         tmp_ecc_params->data_codewords_in_G2 * tmp_ecc_params->num_blocks_in_G2;

        if (data_codewords * byte_len > input_length)
        {
            version_index = i;
            break;
        }
    }
    return version_index;
}

bool QREncoder::estimateVersion(const int& input_length, vector<int>& possible_version)
{
    possible_version.clear();
    if (input_length > version_capacity_database[40].ec_level[ecc_level].encoding_modes[1])
        return false;
    if (input_length <= version_capacity_database[9].ec_level[ecc_level].encoding_modes[3])
    {
        possible_version.push_back(1);
    }
    else if (input_length <= version_capacity_database[9].ec_level[ecc_level].encoding_modes[1])
    {
        possible_version.push_back(1);
        possible_version.push_back(2);
    }
    else if (input_length <= version_capacity_database[26].ec_level[ecc_level].encoding_modes[3])
    {
        possible_version.push_back(2);
    }
    else if (input_length <= version_capacity_database[26].ec_level[ecc_level].encoding_modes[1])
    {
        possible_version.push_back(2);
        possible_version.push_back(3);
    }
    else
    {
        possible_version.push_back(3);
    }
    return true;
}

int QREncoder::versionAuto(const std::string& input_str)
{
    vector<int> possible_version;
    estimateVersion((int)input_str.length(),possible_version);
    int tmp_version = 0;
    vector<uint8_t> payload_tmp;
    int version_range[5] = {0,1,10,27,41};
    for(size_t i = 0; i < possible_version.size(); i++)
    {
        int version_range_index = possible_version[i];
        if (version_range_index == 1)
        {
            tmp_version = 1;
        }
        else if (version_range_index == 2)
        {
            tmp_version = 10;
        }
        else
        {
            tmp_version = 27;
        }
        encodeAuto(input_str, payload_tmp);
        tmp_version = findVersionCapacity((int)payload_tmp.size(), ecc_level,
                                version_range[version_range_index], version_range[version_range_index+1]);
        if(tmp_version != -1)
            break;
    }
    return tmp_version;
}

void QREncoder::init(int mode, int version = 0, int ecc = 0, int structure_num = 2)
{
    ecc_level = ecc;
    version_level = version;
    mode_type = mode;
    struct_num = ( mode == QR_MODE_STRUCTURE ? structure_num:1);
}

void QREncoder::generateQR(const std::string &input)
{
    int v = version_level;
    if (struct_num > 1)
    {
        parity = 0;
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
    int segment_len = (int) ceil((int) input.length() / struct_num);
    Mat qrcode;
    for (int i = 0; i < struct_num; i++)
    {
        sequence_num = (uint8_t) i;
        int segment_begin = i * segment_len;
        int segemnt_end = min((i + 1) * segment_len, (int) input.length()) - 1;
        input_info = input.substr(segment_begin, segemnt_end - segment_begin + 1);
        version_level = (v > 0 ? v : versionAuto(input_info));
        payload.clear();
        payload.reserve(max_payload_len);
        format = Mat(Size(1, 15), CV_8UC1, Scalar(255));
        version_reserved = Mat(Size(1, 18), CV_8UC1, Scalar(255));
        version_size = (21 + (version_level - 1) * 4);
        version_info = &version_info_database[version_level];
        cur_ecc_params = &version_info->ecc[ecc_level];
        original = Mat(Size(version_size, version_size), CV_8UC1, Scalar(255));
        masked_data = original.clone();
        qrcode = masked_data.clone();
        generatingProcess(qrcode);
        final_qrcodes.push_back(qrcode);
    }
}

void QREncoder::formatGenerate(const int& mask_type_num , Mat& format_array)
{
    Mat Polynomial = Mat(Size(1, max_format_length), CV_8UC1, Scalar(0));
    std::string mask_type_bin = decToBin(mask_type_num, 3);
    std::string ec_level_bin = decToBin(eccLevelToCode(ecc_level), 2);
    std::string format_bits = ec_level_bin + mask_type_bin;
    const int level_mask_info_bits = 5;
    const int ecc_info_bits = 10;
    Mat binary_bit = (Mat_<uint8_t >(1, level_mask_info_bits) <<
                                         format_bits[4] - '0', format_bits[3] - '0', format_bits[2] - '0', format_bits[1] - '0', format_bits[0] - '0');
    Mat shift = Mat(Size(ecc_info_bits, 1), CV_8UC1, Scalar(0));
    hconcat(shift, binary_bit, Polynomial);
    Mat format_generator = (Mat_<uint8_t >(1, 11) << 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1);
    Mat ecc_code = gfPolyDiv(Polynomial, format_generator, ecc_info_bits);
    hconcat(ecc_code, binary_bit, format_array);
    Mat system_mask = (Mat_<uint8_t >(1, max_format_length) << 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1);
    for(int i = 0; i < max_format_length; i++)
    {
        format_array.at<uint8_t>(0, i) ^= system_mask.at<uint8_t>(0, i);
    }
}

void QREncoder::versionInfoGenerate(const int& version_level_num, Mat& version_array)
{
    Mat Polynomial = Mat(Size(1, max_version_length), CV_8UC1, Scalar(0));
    std::string version_bits = decToBin(version_level_num, 6);
    Mat binary_bit = (Mat_<uint8_t >(1,6) <<
                                         version_bits[5] - '0', version_bits[4] - '0', version_bits[3] - '0',
                                         version_bits[2] - '0', version_bits[1] - '0', version_bits[0] - '0');
    Mat shift = Mat(Size(12,1), CV_8UC1, Scalar(0));
    hconcat(shift, binary_bit, Polynomial);
    Mat format_generator = (Mat_<uint8_t >(1, 13) << 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1);
    Mat ecc_code = gfPolyDiv(Polynomial, format_generator, 12);
    hconcat(ecc_code, binary_bit, version_array);
}

bool QREncoder::encodeAlpha(const std::string& input, vector<uint8_t>& output)
{
    std::string alpha_map =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
    int bits = 13;
    if (version_level < 10)
        bits = 9;
    else if (version_level < 27)
        bits = 11;
    std::string mode_bits = decToBin(QR_MODE_ALPHA,4);
    loadString(mode_bits, output, true);
    int str_len = int(input.length());
    std::string counter = decToBin(str_len, bits);
    loadString(counter, output, true);
    for (int i = 0; i < str_len - 1; i += 2)
    {
        int index_1 = (int)alpha_map.find(input[i]);
        int index_2 = (int)alpha_map.find(input[i + 1]);
        if(i + 1 >= str_len)
            break;
        if(index_1 == -1 || (index_2 == -1 && i + 1 < str_len))
            return false;
        int result = index_1 * 45 + index_2;
        std::string per_byte = decToBin(result, 11);
        loadString(per_byte, output, true);
    }
    if (str_len % 2 != 0)
    {
        int index = (int)alpha_map.find(*input.rbegin());
        if(index == -1)
            return false;
        std::string per_byte = decToBin(index, 6);
        loadString(per_byte, output, true);
    }
    return true;
}

bool QREncoder::encodeByte(const std::string& input, vector<uint8_t>& output)
{
    int bits = 8;
    if (version_level > 9)
        bits = 16;
    std::string mode_bits = decToBin(QR_MODE_BYTE, 4);
    loadString(mode_bits, output, true);
    int str_len = int(input.length());
    std::string counter = decToBin(str_len, bits);
    loadString(counter, output, true);
    for (int i = 0; i < str_len; i++)
    {
        std::string per_byte = decToBin(int(input[i]), 8);
        loadString(per_byte, output, true);
    }
    return true;
}

bool QREncoder::encodeNumeric(const std::string& input,vector<uint8_t>& output)
{
    int bits = 10;
    if (version_level >= 27)
        bits = 14;
    else if (version_level >= 10)
        bits = 12;
    std::string mode_bits = decToBin(QR_MODE_NUM, 4);
    loadString(mode_bits, output, true);
    int str_len = int(input.length());
    std::string counter = decToBin(str_len, bits);
    loadString(counter, output, true);
    int count = 0;
    while (count + 3 <= str_len)
    {
        if (input[count] > '9' || input[count] < '0' ||
            input[count+1] > '9' || input[count + 1] < '0' ||
            input[count+2] > '9' || input[count + 2] < '0')
            return false;
        int num = 100 * (int)(input[count] - '0') +
                  10 * (int)(input[count + 1] - '0') +
                  (int)(input[count + 2] - '0');
        std::string numeric_group = decToBin(num, 10);
        loadString(numeric_group, output, true);
        count += 3;
    }
    if (count + 2 == str_len)
    {
        if (input[count] > '9' || input[count] < '0'||
            input[count + 1] >'9'|| input[count + 1] < '0')
            return false;
        int num = 10 * (int)(input[count] - '0') +
                  (int)(input[count + 1] - '0');
        std::string numeric_group = decToBin(num, 7);
        loadString(numeric_group, output, true);
    }
    else if (count + 1 == str_len)
    {
        if (input[count] > '9' || input[count] < '0')
            return false;
        int num = (int)(input[count] - '0');
        std::string numeric_group = decToBin(num, 4);
        loadString(numeric_group, output, true);
    }
    return true;
}

bool QREncoder::encodeStructure(const std::string& input, vector<uint8_t>& output)
{
    std::string mode_bits = decToBin(QR_MODE_STRUCTURE, 4);
    loadString(mode_bits, output, true);
    std::string sequence_bits = decToBin(sequence_num, 4);
    loadString(sequence_bits, output, true);
    std::string total_bits = decToBin(total_num, 4);
    loadString(total_bits, output, true);
    std::string parity_bits = decToBin(parity, 8);
    loadString(parity_bits, output, true);
    encodeAuto(input, output);
    return true;
}

bool QREncoder::generateBlock(const std::string& input, int mode, struct autoEncodePerBlock& block)
{
    block.block_load_len = 0;
    block.encoding_mode = mode;
    block.block_load.clear();
    bool result = true;
    switch (mode)
    {
        case QR_MODE_NUM:
            result = encodeNumeric(input, block.block_load);
            break;
        case QR_MODE_ALPHA:
            result = encodeAlpha(input, block.block_load);
            break;
        case QR_MODE_BYTE:
            result = encodeByte(input, block.block_load);
            break;
    }
    block.block_load_len = (int)block.block_load.size();
    return result;
}

struct encodingMethods
{
    int len;
    vector<autoEncodePerBlock> blocks;
    encodingMethods()
    {
        len = 0;
        blocks.clear();
    }
    int sum_len()
    {
        int bits_len = 0;
        for (size_t i = 0; i < blocks.size(); i ++)
        {
            bits_len += (int)blocks[i].block_load.size();
        }
        return bits_len;
    }
};

bool QREncoder::encodeAuto(const std::string& input, vector<uint8_t>& output)
{
    std::string mode_char_set[2];
    mode_char_set[0] = "0123456789";
    mode_char_set[1] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
    vector<encodingMethods> strategy;
    autoEncodePerBlock last_method;
    encodingMethods head;
    strategy.push_back(head);
    size_t len = input.length();
    std::string cur_string = "";
    for (size_t i = 0; i < len; i++)
    {
        cur_string += char (input[i]);
        if (strategy.size() == 1)
        {
            encodingMethods tmp;
            if ((int)mode_char_set[0].find(input[i]) != -1)
            {
                generateBlock(cur_string, QR_MODE_NUM, last_method);
            }
            else if ((int)mode_char_set[1].find(input[i]) != -1)
            {
                generateBlock(cur_string, QR_MODE_ALPHA, last_method);
            }
            else
            {
                generateBlock(cur_string, QR_MODE_BYTE, last_method);
            }
            tmp.blocks.push_back(last_method);
            tmp.len = tmp.sum_len();
            strategy.push_back(tmp);
        }
        else
        {
            size_t str_len = cur_string.length();
            encodingMethods previous;
            encodingMethods new_method;
            new_method.len = error_mode_occur;
            for (size_t j = 0; j < str_len; j++)
            {
                previous = strategy[j];
                std::string sub_string = cur_string.substr(j, str_len - j);
                const int blocks_num = 3;
                autoEncodePerBlock blocks[blocks_num];
                if (!generateBlock(sub_string, QR_MODE_NUM, blocks[0]))
                {
                    blocks[0].block_load_len = error_mode_occur;
                }
                if (!generateBlock(sub_string, QR_MODE_ALPHA, blocks[1]))
                {
                    blocks[1].block_load_len = error_mode_occur;
                }
                generateBlock(sub_string, QR_MODE_BYTE, blocks[2]);
                int index = 0;
                int min_len = error_mode_occur;
                for (int tmp_index = 0; tmp_index < blocks_num; tmp_index++)
                {
                    if (blocks[tmp_index].block_load_len + previous.len < min_len)
                    {
                        index = tmp_index;
                        min_len = blocks[tmp_index].block_load_len + previous.len;
                    }
                }
                previous.blocks.push_back(blocks[index]);
                previous.len = previous.sum_len();

                if(previous.len < new_method.len)
                {
                    new_method = previous;
                }
            }
            strategy.push_back(new_method);
        }
    }
    encodingMethods result = strategy[strategy.size() - 1];
    for (size_t i = 0; i < result.blocks.size(); i++)
    {
        for (int j = 0; j < result.blocks[i].block_load_len; j++)
        {
            output.push_back(result.blocks[i].block_load[j]);
        }
    }
    return true;
}

void QREncoder::padBitStream(){
    int total_data = version_info->total_codewords -
                     cur_ecc_params->ecc_codewords * (cur_ecc_params->num_blocks_in_G1 + cur_ecc_params->num_blocks_in_G2);
    const int bits = 8;
    total_data *= bits;
    int pad_num = total_data - (int)payload.size();

    if (pad_num <= 0)
        return;
    else if (pad_num <= 4)
    {
        std::string pad = decToBin(0, (int)payload.size());
        loadString(pad, payload, true);
    }
    else
    {
        loadString("0000", payload, true);
        int i = payload.size() % bits;

        if (i != 0)
        {
            std::string pad = decToBin(0, bits - i);
            loadString(pad, payload, true);
        }
        pad_num = total_data - (int)payload.size();

        if (pad_num > 0)
        {
            std::string pad_pattern[2] = {"11101100", "00010001"};
            int num = pad_num / bits;
            for (int j = 0; j < num; j++)
            {
                loadString(pad_pattern[j % 2], payload, true);
            }
        }
    }
}

bool QREncoder::stringToBits()
{
    switch (mode_type)
    {
        case QR_MODE_NUM:
            return encodeNumeric(input_info, payload);
        case QR_MODE_ALPHA:
            return encodeAlpha(input_info, payload);
        case QR_MODE_STRUCTURE:
            return encodeStructure(input_info, payload);
        case QR_MODE_BYTE:
            return encodeByte(input_info, payload);
        default:
            return encodeAuto(input_info, payload);
    }
};

void QREncoder::eccGenerate(vector<Mat>& data_blocks, vector<Mat>& ecc_blocks)
{
    int EC_codewords = cur_ecc_params->ecc_codewords;
    int pay_index = 0;
    Mat G_x = polyGenerator(EC_codewords);
    int blocks = cur_ecc_params->num_blocks_in_G2 + cur_ecc_params->num_blocks_in_G1;
    for (int i = 0; i < blocks; i++)
    {
        Mat Block_i, ecc_i;
        int block_len = 0;

        if (i < cur_ecc_params->num_blocks_in_G1)
        {
            block_len = cur_ecc_params->data_codewords_in_G1;
        }
        else
        {
            block_len = cur_ecc_params->data_codewords_in_G2;
        }
        Block_i = Mat(Size(block_len, 1), CV_8UC1, Scalar(0));

        for (int j = 0 ; j < block_len; j++)
        {
            Block_i.at<uint8_t>(0, block_len - 1 - j) = (uchar)getBits(8, payload, pay_index);
        }
        Mat dividend;
        Mat shift = Mat(Size(EC_codewords, 1), CV_8UC1, Scalar(0));
        hconcat(shift, Block_i, dividend);
        ecc_i = gfPolyDiv(dividend, G_x, EC_codewords);
        data_blocks.push_back(Block_i);
        ecc_blocks.push_back(ecc_i);
    }
}

void QREncoder::rearrangeBlocks(const vector<Mat>& data_blocks,const vector<Mat>& ecc_blocks)
{
    rearranged_data.clear();
    rearranged_data.reserve(max_payload_len);
    int blocks = cur_ecc_params->num_blocks_in_G2 + cur_ecc_params->num_blocks_in_G1;
    int col_border = max(cur_ecc_params->data_codewords_in_G2, cur_ecc_params->data_codewords_in_G1);
    int total_codeword_num = version_info->total_codewords;
    int is_not_equal = cur_ecc_params->data_codewords_in_G2 - cur_ecc_params->data_codewords_in_G1;
    for (int i = 0; i < total_codeword_num; i++)
    {
        int cur_col = i / blocks ;
        int cur_row = i % blocks ;
        int data_col = data_blocks[cur_row].cols - 1;
        int ecc_col = ecc_blocks[cur_row].cols - 1;
        std::string bits;
        uint8_t tmp = 0;

        if (cur_col < col_border)
        {
            if (is_not_equal && cur_col == cur_ecc_params->data_codewords_in_G2 - 1 && cur_row < cur_ecc_params->num_blocks_in_G1)
            {
                continue;
            }
            else
            {
                bits = decToBin(data_blocks[cur_row].at<uint8_t>(0, data_col - cur_col), 8);
                tmp = data_blocks[cur_row].at<uint8_t>(0, data_col - cur_col);
            }
        }
        else
        {
            int index = ecc_col - (cur_col - col_border);
            bits = decToBin(ecc_blocks[cur_row].at<uint8_t>(0, index), 8);
            tmp = ecc_blocks[cur_row].at<uint8_t>(0, index);
        }
        rearranged_data.push_back(tmp);
    }
    const int remainder_len []= {0,
                                 0, 7, 7, 7, 7, 7, 0, 0, 0, 0,
                                 0, 0, 0, 3, 3, 3, 3, 3, 3, 3,
                                 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
                                 3, 3, 3, 3, 0, 0, 0, 0, 0, 0};
    int cur_remainder_len = remainder_len[version_level];
    if (cur_remainder_len != 0)
    {
        rearranged_data.push_back(0);
    }
}

void QREncoder::findAutoMaskType()
{
    int best_index = 0;
    int lowest_penalty = INT_MAX;
    int penalty_two_value = 3, penalty_three_value = 40;
    for (int cur_type = 0; cur_type < 8; cur_type++)
    {
        Mat test_result = masked_data.clone();
        Mat test_format = format.clone();
        maskData(cur_type, test_result);
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

void QREncoder::maskData(const int& mask_type_num, Mat& masked)
{
    for (int i = 0; i < version_size; i++)
    {
        for (int j = 0; j < version_size; j++)
        {
            if (original.at<uint8_t>(i, j) == invalid_region_value)
            {
                continue;
            }
            else if((mask_type_num == 0 && !((i + j) % 2)) ||
                    (mask_type_num == 1 && !(i % 2)) ||
                    (mask_type_num == 2 && !(j % 3)) ||
                    (mask_type_num == 3 && !((i + j) % 3 )) ||
                    (mask_type_num == 4 && !(((i / 2) + (j / 3)) % 2)) ||
                    (mask_type_num == 5 && !((i * j) % 2 + (i * j) % 3))||
                    (mask_type_num == 6 && !(((i * j) % 2 + (i * j) % 3) % 2))||
                    ((mask_type_num == 7 && !(((i * j) % 3 + (i + j) % 2) % 2))))
            {
                masked.at<uint8_t>(i, j) = original.at<uint8_t>(i, j) ^ 255;
            }
            else
            {
                masked.at<uint8_t>(i, j) = original.at<uint8_t>(i, j);
            }
        }
    }
}

void QREncoder::writeReservedArea()
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
            for (int i = -5; i <= 5; i ++)
            {
                for (int j = -5; j <= 5; j ++)
                {
                    if (x + i < 0 || x + i >= version_size || y + j < 0 || y + j >= version_size)
                    {
                        continue;
                    }
                    if (!(abs(j) == 2 && -2 <= i && i <=2) &&
                       !(-2 <= j && j <= 2 && abs(i) == 2) &&
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
                        original.at<uint8_t>(x + i, y + j) = invalid_region_value;
                    }
                }
            }

        }
    }
    int x = locator_position[1] - 4;
    int y = locator_position[0] + 5;
    masked_data.at<uint8_t>(x, y) = 0;
    original.at<uint8_t>(x, y) = invalid_region_value;
    if (version_level >= 7)
    {
        for (int i = 0; i <= 6; i++)
        {
            for (int j = version_size - 11; j <= version_size - 8; j++)
            {
                original.at<uint8_t>(i, j) = invalid_region_value;
                original.at<uint8_t>(j, i) = invalid_region_value;
            }
        }
    }
    for (int i = 0; i < version_size; i++)
    {
        for (int j = 0; j < version_size; j++)
        {
            if (original.at<uint8_t>(i, j) == invalid_region_value)
            {
                continue;
            }
            if ((i == 6 || j == 6))
            {
                original.at<uint8_t>(i, j) = invalid_region_value;
                if (!((i == 6) && (j - 7) % 2 == 0) &&
                    !((j == 6) && ((i - 7) % 2 == 0)))
                {
                    masked_data.at<uint8_t>(i, j) = 0;
                }
            }
        }
    }
    for (int first_coord = 0; first_coord < max_alignment && version_info->alignment_pattern[first_coord]; first_coord++)
    {
        for (int second_coord = 0; second_coord < max_alignment && version_info->alignment_pattern[second_coord]; second_coord++)
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
                        original.at<uint8_t>(x + i, y + j) = invalid_region_value;
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

bool QREncoder::writeBit(int x, int y, bool value)
{
    if (original.at<uint8_t>(y, x) == invalid_region_value)
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

void QREncoder::writeData()
{
    int y = version_size - 1;
    int x = version_size - 1;
    int dir = -1;
    int count = 0;
    int codeword_value = rearranged_data[0];
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
            if (count % 8 == 0)
            {
                codeword_value = rearranged_data[count / 8];
            }
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

void QREncoder::fillReserved(const Mat& format_array, Mat& masked)
{
    for (int i = 0; i < 7; i++)
    {
        if (format_array.at<uint8_t>(0, max_format_length - 1 - i) == 0)
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
        if (format_array.at<uint8_t>(0, max_format_length - 1 - (7 + i)) == 0)
        {
            masked.at<uint8_t>(8, version_size - 8 + i) = 255;
        }
        else
        {
            masked.at<uint8_t>(8, version_size - 8 + i) = 0;
        }
    }
    static const int xs_format[max_format_length] = {
            8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
    };
    static const int ys_format[max_format_length] = {
            0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
    };
    for (int i = max_format_length - 1; i >= 0; i--)
    {
        if (format_array.at<uint8_t>(0, i) == 0)
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
        const int xs_version[version_block_width][max_version_length] = {
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
        const int ys_version[version_block_width][max_version_length] = {
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
            for (int j = 0; j < max_version_length; j++)
            {
                if (version_reserved.at<uint8_t>(0, max_version_length - j - 1) == 0)
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

void QREncoder::structureFinalMessage()
{
    writeReservedArea();
    writeData();
    findAutoMaskType();
    maskData(mask_type, masked_data);
    formatGenerate(mask_type, format);
    versionInfoGenerate(version_level, version_reserved);
    fillReserved(format, masked_data);
}

void QREncoder::generatingProcess(Mat& final_result)
{
    vector<Mat> data_blocks, ecc_blocks;
    if (!stringToBits())
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

struct QRCodeEncoder::Impl
{
public:
    Impl() {}
    ~Impl() {}
};
QRCodeEncoder::QRCodeEncoder() : p(new Impl) {}
QRCodeEncoder::~QRCodeEncoder() {}

bool QRCodeEncoder::generate(cv::String input, cv::OutputArray output,
                             int version, int correction_level, int mode,
                             int structure_number, Size output_size)
{
    int output_type = output.kind();
    QREncoder my_qrcode;
    my_qrcode.init(mode, version, correction_level, structure_number);
    my_qrcode.generateQR(input);
    if (my_qrcode.final_qrcodes.size() == 0)
        return false;
    vector<Mat> result = my_qrcode.final_qrcodes;
    if (output_type == _InputArray::STD_VECTOR_MAT)
    {
        output.create((int)result.size(), 1, result[0].type());
        std::vector<Mat> dst;
        output.getMatVector(dst);
        for (int i = 0; i < (int)result.size(); i++)
        {
            Mat cur_mat = result[i];
            if(cur_mat.cols < output_size.height)
                resize(cur_mat, cur_mat, output_size, 0, 0, INTER_AREA);
            output.getMatRef(i) = cur_mat;
        }
    }
    else if (output_type == _InputArray::MAT)
    {
        Mat cur_mat = my_qrcode.final_qrcodes[0];
        if (cur_mat.cols < output_size.height)
            resize(cur_mat, cur_mat, output_size, 0, 0, INTER_AREA);
        output.assign(cur_mat);
    }
    else
    {
        return false;
    }
    return true;
}
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "precomp.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"

#ifdef HAVE_QUIRC
#include "quirc.h"
#endif

#include <limits>
#include <cmath>
#include <iostream>
#include <queue>
#include <limits>
#include <iomanip>

namespace cv
{

/* Limits on the maximum size of QR-codes and their content. */
#define  MAX_BITMAP	3917
#define  MAX_PAYLOAD	8896

#define FORMAT_LENGTH 15
#define  MAX_VERSION     40
#define  MAX_ALIGNMENT   7
#define MAX_POLY       64

#define MAX_VERSION     40
#define MAX_ALIGNMENT   7

#define INVALID_REGION 110 /*for the reserved value when reading the data*/
#define  CODEWORD_LEN 8

enum OUTPUT{
    HEX,OCT,ALPHA
};
/**
 * Encoding mode.
 */
typedef enum {
    QR_MODE_NUL        = 0b0000,   ///< Terminator (NUL character). Internal use only
    QR_MODE_ECI        = 0b0111,        ///< ECI mode
    QR_MODE_NUM        = 0b0001,    ///< Numeric mode
    QR_MODE_ALPHA      = 0b0010,         ///< Alphabet-numeric mode
    QR_MODE_BYTE       = 0b0100,          ///< 8-bit data mode
    QR_MODE_KANJI      = 0b1000,      ///< Kanji (shift-jis) mode
    QR_MODE_STRUCTURE  = 0b0011,  ///< Internal use only
    QR_MODE_FNC1FIRST  = 0b0101, ///< FNC1, first position
    QR_MODE_FNC1SECOND = 0b1001, ///< FNC1, second position
} QRencodeMode;

using std::vector;
using std::cout;
using std::endl;

std::string D2B(uint16_t my_format);
int ecc_code2level(int code);

uint8_t gf_pow(uint8_t x , int power);
uint8_t gf_inverse(uint8_t x);
uint8_t gf_mul(const uint8_t &x,const uint8_t& y);
uint8_t gf_div(const uint8_t &x,const uint8_t& y);

uint8_t gf_poly_eval(const Mat& poly,uint8_t x);
Mat gf_poly_scale(const Mat & p,int scalar);
Mat gf_poly_add(const Mat & p,const Mat & q);
Mat gf_poly_mul(const Mat &p,const Mat &q);
std::string show_poly(const Mat& p,OUTPUT o);
Mat gf_poly_div(const Mat& dividend,const Mat& divisor,const int& ecc_num);

Mat find_error_locator(const vector<uint8_t>&synd,int & errors_len);
vector<int > find_errors(const Mat& sigma,const int &errors_len,const int & msg_len);
Mat error_correct(const Mat & msg_in ,const vector<uint8_t>&synd,const Mat & e_loc_poly,const vector<int> &error_index);


int hamming_weight(uint16_t x);
int hamming_detect(uint16_t fmt);

int block_syndromes(const Mat & block, int synd_num,vector<uint8_t>& synd);
int get_bits(const int& bits,uint8_t * & ptr);

/*total codewords are divided into two groups
 *The ecc_codewords are the same in two groups*/
struct block_params {
    int ecc_codewords;           //Number of error correction blocks
    int num_blocks_in_G1;
    int data_codewords_in_G1;
    int num_blocks_in_G2;
    int data_codewords_in_G2;
};

struct version_info {
    int	total_codewords;
    int	apat[MAX_ALIGNMENT];  //(location of alignment pattern)
    block_params ecc[4];
};

const version_info version_db[MAX_VERSION + 1] = {
        { /* Version 0 */
                0,
                {0,0,0,0,0,0,0},
                {
                        {0	,0	,0,0,0},
                        {0	,0	,0,0,0},
                        {0	,0	,0,0,0},
                        {0	,0	,0 ,0,0}
                }
        },
        { /* Version 1 */
                26,
                {0,0,0,0,0,0,0},
                {
                        {7	,1	,19,0,0},
                        {10	,1	,16,0,0},
                        {13	,1	,13,0,0},
                        {17	,1	,9 ,0,0}
                }
        },
        { /* Version 2 */
                44,
                {6, 18, 0,0,0,0,0},
                {
                        { 10,	1,	34,0,0},
                        {  16,	1,	28,0,0},
                        {  22,	1,	22,0,0},
                        {  28,	1,	16,0,0}
                }
        },
        { /* Version 3 */
                70,
                {6, 22, 0,0,0,0,0},
                {
                        {  15,	1,	55,0,0},
                        {  26,	1,	44,0,0},
                        {  18,	2,	17,0,0},
                        {  22,	2,	13,0,0}
                }
        },
        { /* Version 4 */
                100,
                {6, 26, 0,0,0,0,0},
                {
                        {  20,	1,	80,0,0},
                        {  18,	2,	32,0,0},
                        {  26,	2,	24,0,0},
                        {  16,	4,	9 ,0,0}
                }
        },
        { /* Version 5 */
                134,
                {6, 30, 0,0,0,0,0},
                {
                        {  26,	1,	108,0,  0},
                        {  24,	2,	43, 0,  0},
                        {  18,	2,	15,	2,	16},
                        {  22,	2,	11,	2,	12}
                }
        },
        { /* Version 6 */
                172,
                {6, 34, 0,0,0,0,0},
                {
                        {  18,	2,	68,0,0},
                        {  16,	4,	27,0,0},
                        {  24,	4,	19,0,0},
                        {  28,	4,	15,0,0}
                }
        },
        { /* Version 7 */
                196,
                {6, 22, 38, 0,0,0,0},
                {
                        {  20,	2,	78,0,0},
                        {  18,	4,	31,0,0},
                        {  18,	2,	14,	4,	15},
                        {  26,	4,	13,	1,	14}
                }
        },
        { /* Version 8 */
                242,
                {6, 24, 42, 0,0,0,0},
                {
                        {  24,	2,	97,0,0},
                        {  22,	2,	38,	2,	39},
                        {  22,	4,	18,	2,	19},
                        {  26,	4,	14,	2,	15}
                }
        },
        { /* Version 9 */
                292,
                {6, 26, 46, 0,0,0,0},
                {
                        {  30,	2,	116,0,0},
                        {  22,	3,	36,	2,	37},
                        {  20,	4,	16,	4,	17},
                        {  24,	4,	12,	4,	13}
                }
        },
        { /* Version 10 */
                346,
                {6, 28, 50, 0,0,0,0},
                {
                        {  18,	2,	68,	2,	69},
                        {  26,	4,	43,	1,	44},
                        {  24,	6,	19,	2,	20},
                        {  28,	6,	15,	2,	16}
                }
        },
        { /* Version 11 */
                404,
                {6, 30, 54, 0,0,0,0},
                {
                        {  20,	4,	81, 0,0},
                        {  30,	1,	50,	4,	51},
                        {  28,	4,	22,	4,	23},
                        {  24,	3,	12,	8,	13}
                }
        },
        { /* Version 12 */
                466,
                {6, 32, 58, 0,0,0,0},
                {
                        {  24,	2,	92,	2,	93},
                        {  22,	6,	36,	2,	37},
                        {  26,	4,	20,	6,	21},
                        {  28,	7,	14,	4,	15}
                }
        },
        { /* Version 13 */
                532,
                {6, 34, 62, 0,0,0,0},
                {
                        {  26,	4,	107,0,0},
                        {  22,	8,	37,	1,	38},
                        {  24,	8,	20,	4,	21},
                        {  22,	12,	11,	4,	12}
                }
        },
        { /* Version 14 */
                581,
                {6, 26, 46, 66, 0,0,0},
                {
                        {  30,	3,	115,1,	116},
                        {  24,	4,	40,	5,	41},
                        {  20,	11,	16,	5,	17},
                        {  24,	11,	12,	5,	13}
                }
        },
        { /* Version 15 */
                655,
                {6, 26, 48, 70, 0,0,0},
                {
                        {  22,	5,	87,	1,	88},
                        {  24,	5,	41,	5,	42},
                        {  30,	5,	24,	7,	25},
                        {  24,	11,	12,	7,	13}
                }
        },
        { /* Version 16 */
                733,
                {6, 26, 50, 74, 0,0,0},
                {
                        {  24,	5,	98,	1,	99},
                        {  28,	7,	45,	3,	46},
                        {  24,	15,	19,	2,	20},
                        {  30,	3,	15,	13,	16}
                }
        },
        { /* Version 17 */
                815,
                {6, 30, 54, 78, 0,0,0},
                {
                        { 28,	1,	107,5,	108},
                        { 28,	10,	46,	1,	47},
                        { 28,	1,	22,	15,	23},
                        { 28,	2,	14,	17,	15}
                }
        },
        { /* Version 18 */
                901,
                {6, 30, 56, 82, 0,0,0},
                {
                        {  30,	5,	120,1,	121},
                        {  26,	9,	43,	4,	44},
                        {  28,	17,	22,	1,	23},
                        {  28,	2,	14,	19,	15}
                }
        },
        { /* Version 19 */
                991,
                {6, 30, 58, 86, 0,0,0},
                {
                        {  28,	3,	113,4,	114},
                        {  26,	3,	44,	11,	45},
                        {  26,	17,	21,	4,	22},
                        {  26,	9,	13,	16,	14}
                }
        },
        { /* Version 20 */
                1085,
                {6, 34, 62, 90, 0,0,0},
                {
                        {  28,	3,	107,5,	108},
                        {  26,	3,	41,	13,	42},
                        {  30,	15,	24,	5,	25},
                        {  28,	15,	15,	10,	16}
                }
        },
        { /* Version 21 */
                1156,
                {6, 28, 50, 72, 92, 0,0},
                {
                        {  28,	4,	116,4,  117},
                        {  26,	17,	42, 0,  0},
                        {  28,	17,	22,	6,	23},
                        {  30,	19,	16,	6,	17}
                }
        },
        { /* Version 22 */
                1258,
                {6, 26, 50, 74, 98, 0,0},
                {
                        {  28,	2,	111, 7,	 112},
                        {  28,	17,	46,  0,  0},
                        {  30,	7,	24,	 16, 25},
                        {  24,	34,	13,  0,  0}
                }
        },
        { /* Version 23 */
                1364,
                {6, 30, 54, 78, 102, 0,0},
                {
                        {  30,	4,	121,5,	122},
                        {  28,	4,	47,	14,	48},
                        {  30,	11,	24,	14,	25},
                        {  30,	16,	15,	14,	16}
                }
        },
        { /* Version 24 */
                1474,
                {6, 28, 54, 80, 106, 0,0},
                {
                        {  30,	6,	117,4,	118},
                        {  28,	6,	45,	14,	46},
                        {  30,	11,	24,	16,	25},
                        {  30,	30,	16,	2,	17}
                }
        },
        { /* Version 25 */
                1588,
                {6, 32, 58, 84, 110, 0,0},
                {
                        {  26,	8,	106,4,	107},
                        {  28,	8,	47,	13,	48},
                        {  30,	7,	24,	22,	25},
                        {  30,	22,	15,	13,	16}
                }
        },
        { /* Version 26 */
                1706,
                {6, 30, 58, 86, 114, 0,0},
                {
                        {  28,	10,	114,2,	115},
                        {  28,	19,	46,	4,	47},
                        {  28,	28,	22,	6,	23},
                        {  30,	33,	16,	4,	17}
                }
        },
        { /* Version 27 */
                1828,
                {6, 34, 62, 90, 118, 0,0},
                {
                        {  30,	8,	122,4,	123},
                        {  28,	22,	45,	3,	46},
                        {  30,	8,	23,	26,	24},
                        {  30,	12,	15,	28,	16}
                }
        },
        { /* Version 28 */
                1921,
                {6, 26, 50, 74, 98, 122, 0},
                {
                        {  30,	3,	117,10,	118},
                        {  28,	3,	45,	23,	46},
                        {  30,	4,	24,	31,	25},
                        {  30,	11,	15,	31,	16}
                }
        },
        { /* Version 29 */
                2051,
                {6, 30, 54, 78, 102, 126, 0},
                {
                        {  30,	7,	116,7,	117},
                        {  28,	21,	45,	7,	46},
                        {  30,	1,	23,	37,	24},
                        {  30,	19,	15,	26,	16}
                }
        },
        { /* Version 30 */
                2185,
                {6, 26, 52, 78, 104, 130, 0},
                {
                        {  30,	5,	115,10,	116},
                        {  28,	19,	47,	10,	48},
                        {  30,	15,	24,	25,	25},
                        {  30,	23,	15,	25,	16}
                }
        },
        { /* Version 31 */
                2323,
                {6, 30, 56, 82, 108, 134, 0},
                {
                        {  30,	13,	115,3,	116},
                        {  28,	2,	46,	29,	47},
                        {  30,	42,	24,	1,	25},
                        {  30,	23,	15,	28,	16}
                }
        },
        { /* Version 32 */
                2465,
                {6, 34, 60, 86, 112, 138, 0},
                {
                        {  30,	17,	115, 0,     0},
                        {  28,	10,	46,	 23,	47},
                        {  30,	10,	24,	 35,	25},
                        {  30,	19,	15,	 35,	16}
                }
        },
        { /* Version 33 */
                2611,
                {6, 30, 58, 86, 114, 142, 0},
                {
                        {  30,	17,	115,1,	116},
                        {  28,	14,	46,	21,	47},
                        {  30,	29,	24,	19,	25},
                        {  30,	11,	15,	46,	16}
                }
        },
        { /* Version 34 */
                2761,
                {6, 34, 62, 90, 118, 146, 0},
                {
                        {  30,	13,	115,6,	116},
                        {  28,	14,	46,	23,	47},
                        {  30,	44,	24,	7,	25},
                        {  30,	59,	16,	1,	17}
                }
        },
        { /* Version 35 */
                2876,
                {6, 30, 54, 78, 102, 126, 150},
                {
                        {  30,	12,	121,7,	122},
                        {  28,	12,	47,	26,	48},
                        {  30,	39,	24,	14,	25},
                        {  30,	22,	15,	41,	16}
                }
        },
        { /* Version 36 */
                3034,
                {6, 24, 50, 76, 102, 128, 154},
                {
                        {  30,	6,	121,14,	122},
                        {  28,	6,	47,	34,	48},
                        {  30,	46,	24,	10,	25},
                        {  30,	2,	15,	64,	16}
                }
        },
        { /* Version 37 */
                3196,
                {6, 28, 54, 80, 106, 132, 158},
                {
                        {  30,	17,	122,4,	123},
                        {  28,	29,	46,	14,	47},
                        {  30,	49,	24,	10,	25},
                        {  30,	24,	15,	46,	16}
                }
        },
        { /* Version 38 */
                3362,
                {6, 32, 58, 84, 110, 136, 162},
                {
                        {  30,	4,	122,18,	123},
                        {  28,	13,	46,	32,	47},
                        {  30,	48,	24,	14,	25},
                        {  30,	42,	15,	32,	16}
                }
        },
        { /* Version 39 */
                3532,
                {6, 26, 54, 82, 110, 138, 166},
                {
                        {  30,	20,	117,4,	118},
                        {  28,	40,	47,	7,	48},
                        {  30,	43,	24,	22,	25},
                        {  30,	10,	15,	67,	16}
                }
        },
        { /* Version 40 */
                3706,
                {6, 30, 58, 86, 114, 142, 170},
                {
                        {  30,	19,	118,6,	119},
                        {  28,	18,	47,	31,	48},
                        {  30,	34,	24,	34,	25},
                        {  30,	20,	15,	61,	16}
                }
        }
};
static const uint16_t after_mask_format [32]={
        0x5412,0x5125,0x5e7c,0x5b4b,0x45f9,  0x40ce,0x4f97,0x4aa0,0x77c4,0x72f3,
        0x7daa,0x789d,0x662f,0x6318,0x6c41,  0x6976,0x1689,0x13be,0x1ce7,0x19d0,
        0x0762,0x0255,0x0d0c,0x083b,0x355f,  0x3068,0x3f31,0x3a06,0x24b4,0x2183,
        0x2eda,0x2bed
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

/* This enum describes the various decoder errors which may occur. */
typedef enum {
    SUCCESS = 0,
    ERROR_INVALID_GRID_SIZE,
    ERROR_INVALID_VERSION,
    ERROR_FORMAT_ECC,
    ERROR_DATA_ECC,
    ERROR_UNKNOWN_DATA_TYPE,
    ERROR_DATA_OVERFLOW,
    ERROR_DATA_UNDERFLOW
}  decode_error ;
/*
 * params @ ecc_level
 * func   @ make the ecc_level more direct
 * return @ the actual value of ecc_level
 * */
int ecc_code2level(int code){
    switch (code){
        case 0b01://L	01
            return 0;
        case 0b00 ://M	00
            return 1;
        case 0b11 ://Q	11
            return 2;
        case 0b10 ://H	10
            return 3;
    }
    return 0;
}
/*
 * params @ a number
 * func   @ convert a dec to bin
 * return @ a bin string for print
 * */

std::string D2B(uint16_t my_format){
    std::string f;
    for(int i=my_format;i>0;i=i>>1){
        if(i%2==1)
            f='1'+f;
        else
            f='0'+f;
    }
    return f;
}

/*hamming_weight:
 *      get the distance by counting the number of 1
 * params @ uint16_t x(input the XOR of two binary string)
 * return @ the distance
 */
int hamming_weight(uint16_t x){
    int weight=0;
    while(x>0){
        weight += x & 1;
        x >>= 1;
    }
    return weight;
}

/*hamming_detect:
 *      find the best matched string
 * params @ uint16_t fmt(input format)
 * return @ the index of matched component in the look-up table
*/

int hamming_detect(uint16_t fmt){
    int best_fmt = -1;
    int best_dist = 15;
    int test_dist;
    int index=0;
    for(;index<32;index++){
        /*fetch out from the table*/
        uint16_t test_code=after_mask_format[index];
        test_dist=hamming_weight(fmt ^ test_code);
        /* find the smallest distance*/
        if (test_dist < best_dist){
            best_dist = test_dist;
            best_fmt = index;
        }
            /*can't match two components*/
        else if(test_dist == best_dist) {
            best_fmt = -1;
        }
    }
    cout<<"best_fmt : "<<best_fmt<<" best_dist : "<<best_dist<<endl;
    return best_fmt;
}

static bool checkQRInputImage(InputArray img, Mat& gray)
{
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");

    if (img.cols() <= 20 || img.rows() <= 20)
    {
        return false;  // image data is not enough for providing reliable results
    }
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = img.getMat();
    }
    return true;
}

static void updatePointsResult(OutputArray points_, const vector<Point2f>& points)
{
    if (points_.needed())
    {
        int N = int(points.size() / 4);
        if (N > 0)
        {
            Mat m_p(N, 4, CV_32FC2, (void*)&points[0]);
            int points_type = points_.fixedType() ? points_.type() : CV_32FC2;
            m_p.reshape(2, points_.rows()).convertTo(points_, points_type);  // Mat layout: N x 4 x 2cn
        }
        else
        {
            points_.release();
        }
    }
}



class QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints();
    Mat getBinBarcode() { return bin_barcode; }
    Mat getStraightBarcode() { return straight_barcode; }
    vector<Point2f> getTransformationPoints() { return transformation_points; }
    static Point2f intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2);
protected:
    vector<Vec3d> searchHorizontalLines();
    vector<Point2f> separateVerticalLines(const vector<Vec3d> &list_lines);
    vector<Point2f> extractVerticalLines(const vector<Vec3d> &list_lines, double eps);
    void fixationPoints(vector<Point2f> &local_point);
    vector<Point2f> getQuadrilateral(vector<Point2f> angle_list);
    bool testBypassRoute(vector<Point2f> hull, int start, int finish);
    inline double getCosVectors(Point2f a, Point2f b, Point2f c);

    Mat barcode, bin_barcode, resized_barcode, resized_bin_barcode, straight_barcode;
    vector<Point2f> localization_points, transformation_points;
    double eps_vertical, eps_horizontal, coeff_expansion;
    enum resize_direction { ZOOMING, SHRINKING, UNCHANGED } purpose;
};


void QRDetect::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!src.empty());
    barcode = src.clone();
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        purpose = ZOOMING;
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        const int width  = cvRound(src.size().width  / coeff_expansion);
        const int height = cvRound(src.size().height  / coeff_expansion);
        Size new_size(width, height);
        resize(src, resized_barcode, new_size, 0, 0, INTER_AREA);
    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;

    if (!barcode.empty())
        adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    else
        bin_barcode.release();

    if (!resized_barcode.empty())
        adaptiveThreshold(resized_barcode, resized_bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    else
        resized_bin_barcode.release();
}

vector<Vec3d> QRDetect::searchHorizontalLines()
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    const int height_bin_barcode = bin_barcode.rows;
    const int width_bin_barcode  = bin_barcode.cols;
    const size_t test_lines_size = 5;
    double test_lines[test_lines_size];
    vector<size_t> pixels_position;

    for (int y = 0; y < height_bin_barcode; y++)
    {
        pixels_position.clear();
        const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);

        int pos = 0;
        for (; pos < width_bin_barcode; pos++) { if (bin_barcode_row[pos] == 0) break; }
        if (pos == width_bin_barcode) { continue; }

        pixels_position.push_back(pos);
        pixels_position.push_back(pos);
        pixels_position.push_back(pos);

        uint8_t future_pixel = 255;
        for (int x = pos; x < width_bin_barcode; x++)
        {
            if (bin_barcode_row[x] == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                pixels_position.push_back(x);
            }
        }
        pixels_position.push_back(width_bin_barcode - 1);
        for (size_t i = 2; i < pixels_position.size() - 4; i+=2)
        {
            test_lines[0] = static_cast<double>(pixels_position[i - 1] - pixels_position[i - 2]);
            test_lines[1] = static_cast<double>(pixels_position[i    ] - pixels_position[i - 1]);
            test_lines[2] = static_cast<double>(pixels_position[i + 1] - pixels_position[i    ]);
            test_lines[3] = static_cast<double>(pixels_position[i + 2] - pixels_position[i + 1]);
            test_lines[4] = static_cast<double>(pixels_position[i + 3] - pixels_position[i + 2]);

            double length = 0.0, weight = 0.0;  // TODO avoid 'double' calculations

            for (size_t j = 0; j < test_lines_size; j++) { length += test_lines[j]; }

            if (length == 0) { continue; }
            for (size_t j = 0; j < test_lines_size; j++)
            {
                if (j != 2) { weight += fabs((test_lines[j] / length) - 1.0/7.0); }
                else        { weight += fabs((test_lines[j] / length) - 3.0/7.0); }
            }

            if (weight < eps_vertical)
            {
                Vec3d line;
                line[0] = static_cast<double>(pixels_position[i - 2]);
                line[1] = y;
                line[2] = length;
                result.push_back(line);
            }
        }
    }
    return result;
}

vector<Point2f> QRDetect::separateVerticalLines(const vector<Vec3d> &list_lines)
{
    CV_TRACE_FUNCTION();

    for (int coeff_epsilon = 1; coeff_epsilon < 10; coeff_epsilon++)
    {
        vector<Point2f> point2f_result = extractVerticalLines(list_lines, eps_horizontal * coeff_epsilon);
        if (!point2f_result.empty())
        {
            vector<Point2f> centers;
            Mat labels;
            double compactness = kmeans(
                    point2f_result, 3, labels,
                    TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
                    3, KMEANS_PP_CENTERS, centers);
            if (compactness == 0)
                continue;
            if (compactness > 0)
            {
                return point2f_result;
            }
        }
    }
    return vector<Point2f>();  // nothing
}

vector<Point2f> QRDetect::extractVerticalLines(const vector<Vec3d> &list_lines, double eps)
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    vector<double> test_lines; test_lines.reserve(6);

    for (size_t pnt = 0; pnt < list_lines.size(); pnt++)
    {
        const int x = cvRound(list_lines[pnt][0] + list_lines[pnt][2] * 0.5);
        const int y = cvRound(list_lines[pnt][1]);

        // --------------- Search vertical up-lines --------------- //

        test_lines.clear();
        uint8_t future_pixel_up = 255;

        int temp_length_up = 0;
        for (int j = y; j < bin_barcode.rows - 1; j++)
        {
            uint8_t next_pixel = bin_barcode.ptr<uint8_t>(j + 1)[x];
            temp_length_up++;
            if (next_pixel == future_pixel_up)
            {
                future_pixel_up = static_cast<uint8_t>(~future_pixel_up);
                test_lines.push_back(temp_length_up);
                temp_length_up = 0;
                if (test_lines.size() == 3)
                    break;
            }
        }

        // --------------- Search vertical down-lines --------------- //

        int temp_length_down = 0;
        uint8_t future_pixel_down = 255;
        for (int j = y; j >= 1; j--)
        {
            uint8_t next_pixel = bin_barcode.ptr<uint8_t>(j - 1)[x];
            temp_length_down++;
            if (next_pixel == future_pixel_down)
            {
                future_pixel_down = static_cast<uint8_t>(~future_pixel_down);
                test_lines.push_back(temp_length_down);
                temp_length_down = 0;
                if (test_lines.size() == 6)
                    break;
            }
        }

        // --------------- Compute vertical lines --------------- //

        if (test_lines.size() == 6)
        {
            double length = 0.0, weight = 0.0;  // TODO avoid 'double' calculations

            for (size_t i = 0; i < test_lines.size(); i++)
                length += test_lines[i];

            CV_Assert(length > 0);
            for (size_t i = 0; i < test_lines.size(); i++)
            {
                if (i % 3 != 0)
                {
                    weight += fabs((test_lines[i] / length) - 1.0/ 7.0);
                }
                else
                {
                    weight += fabs((test_lines[i] / length) - 3.0/14.0);
                }
            }

            if (weight < eps)
            {
                result.push_back(list_lines[pnt]);
            }
        }
    }

    vector<Point2f> point2f_result;
    if (result.size() > 2)
    {
        for (size_t i = 0; i < result.size(); i++)
        {
            point2f_result.push_back(
                  Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
                          static_cast<float>(result[i][1])));
        }
    }
    return point2f_result;
}

void QRDetect::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();
    double cos_angles[3], norm_triangl[3];

    norm_triangl[0] = norm(local_point[1] - local_point[2]);
    norm_triangl[1] = norm(local_point[0] - local_point[2]);
    norm_triangl[2] = norm(local_point[1] - local_point[0]);

    cos_angles[0] = (norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[0] * norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[1] * norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1]
                  -  norm_triangl[2] * norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
        local_point.clear();
        return;
    }

    size_t i_min_cos =
       (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
       (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]),
            left_point(local_point[left_index]), right_point(local_point[right_index]),
            central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                                      static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            LineIterator& li = list_line_iter[k];
            uint8_t future_pixel = 255, count_index = 0;
            for(int j = 0; j < li.count; j++, ++li)
            {
                const Point p = li.pos();
                if (p.x >= bin_barcode.cols ||
                    p.y >= bin_barcode.rows)
                {
                    break;
                }

                const uint8_t value = bin_barcode.at<uint8_t>(p);
                if (value == future_pixel)
                {
                    future_pixel = static_cast<uint8_t>(~future_pixel);
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(p);
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos) { std::swap(local_point[0], local_point[index_max]); }
    else { local_point.clear(); return; }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if( determinant(m) > 0 )
    {
        std::swap(local_point[1], local_point[2]);
    }
}

bool QRDetect::localization()
{
    CV_TRACE_FUNCTION();
    Point2f begin, end;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if( list_lines_x.empty() ) { return false; }
    vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
    if( list_lines_y.empty() ) { return false; }

    vector<Point2f> centers;
    Mat labels;
    kmeans(list_lines_y, 3, labels,
           TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
           3, KMEANS_PP_CENTERS, localization_points);

    fixationPoints(localization_points);

    bool suare_flag = false, local_points_flag = false;
    double triangle_sides[3];
    double triangle_perim, square_area, img_square_area;
    if (localization_points.size() == 3)
    {
        triangle_sides[0] = norm(localization_points[0] - localization_points[1]);
        triangle_sides[1] = norm(localization_points[1] - localization_points[2]);
        triangle_sides[2] = norm(localization_points[2] - localization_points[0]);

        triangle_perim = (triangle_sides[0] + triangle_sides[1] + triangle_sides[2]) / 2;

        square_area = sqrt((triangle_perim * (triangle_perim - triangle_sides[0])
                                           * (triangle_perim - triangle_sides[1])
                                           * (triangle_perim - triangle_sides[2]))) * 2;
        img_square_area = bin_barcode.cols * bin_barcode.rows;

        if (square_area > (img_square_area * 0.2))
        {
            suare_flag = true;
        }
    }
    else
    {
        local_points_flag = true;
    }
    if ((suare_flag || local_points_flag) && purpose == SHRINKING)
    {
        localization_points.clear();
        bin_barcode = resized_bin_barcode.clone();
        list_lines_x = searchHorizontalLines();
        if( list_lines_x.empty() ) { return false; }
        list_lines_y = separateVerticalLines(list_lines_x);
        if( list_lines_y.empty() ) { return false; }

        kmeans(list_lines_y, 3, labels,
               TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
               3, KMEANS_PP_CENTERS, localization_points);

        fixationPoints(localization_points);
        if (localization_points.size() != 3) { return false; }

        const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
        const int height = cvRound(bin_barcode.size().height * coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
        for (size_t i = 0; i < localization_points.size(); i++)
        {
            localization_points[i] *= coeff_expansion;
        }
    }
    if (purpose == ZOOMING)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
        for (size_t i = 0; i < localization_points.size(); i++)
        {
            localization_points[i] /= coeff_expansion;
        }
    }

    for (size_t i = 0; i < localization_points.size(); i++)
    {
        for (size_t j = i + 1; j < localization_points.size(); j++)
        {
            if (norm(localization_points[i] - localization_points[j]) < 10)
            {
                return false;
            }
        }
    }
    return true;

}

bool QRDetect::computeTransformationPoints()
{
    CV_TRACE_FUNCTION();
    if (localization_points.size() != 3) { return false; }

    vector<Point> locations, non_zero_elem[3], newHull;
    vector<Point2f> new_non_zero_elem[3];
    for (size_t i = 0; i < 3; i++)
    {
        Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
        uint8_t next_pixel, future_pixel = 255;
        int count_test_lines = 0, index = cvRound(localization_points[i].x);
        for (; index < bin_barcode.cols - 1; index++)
        {
            next_pixel = bin_barcode.ptr<uint8_t>(cvRound(localization_points[i].y))[index + 1];
            if (next_pixel == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                count_test_lines++;
                if (count_test_lines == 2)
                {
                    floodFill(bin_barcode, mask,
                              Point(index + 1, cvRound(localization_points[i].y)), 255,
                              0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                    break;
                }
            }
        }
        Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
        findNonZero(mask_roi, non_zero_elem[i]);
        newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
    }
    convexHull(newHull, locations);
    for (size_t i = 0; i < locations.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < non_zero_elem[j].size(); k++)
            {
                if (locations[i] == non_zero_elem[j][k])
                {
                    new_non_zero_elem[j].push_back(locations[i]);
                }
            }
        }
    }

    double pentagon_diag_norm = -1;
    Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
        {
            double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
            if (temp_norm > pentagon_diag_norm)
            {
                down_left_edge_point = new_non_zero_elem[1][i];
                up_right_edge_point  = new_non_zero_elem[2][j];
                pentagon_diag_norm = temp_norm;
            }
        }
    }

    if (down_left_edge_point == Point2f(0, 0) ||
        up_right_edge_point  == Point2f(0, 0) ||
        new_non_zero_elem[0].size() == 0) { return false; }

    double max_area = -1;
    up_left_edge_point = new_non_zero_elem[0][0];

    for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
    {
        vector<Point2f> list_edge_points;
        list_edge_points.push_back(new_non_zero_elem[0][i]);
        list_edge_points.push_back(down_left_edge_point);
        list_edge_points.push_back(up_right_edge_point);

        double temp_area = fabs(contourArea(list_edge_points));
        if (max_area < temp_area)
        {
            up_left_edge_point = new_non_zero_elem[0][i];
            max_area = temp_area;
        }
    }

    Point2f down_max_delta_point, up_max_delta_point;
    double norm_down_max_delta = -1, norm_up_max_delta = -1;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i])
                               + norm(down_left_edge_point - new_non_zero_elem[1][i]);
        if (norm_down_max_delta < temp_norm_delta)
        {
            down_max_delta_point = new_non_zero_elem[1][i];
            norm_down_max_delta = temp_norm_delta;
        }
    }


    for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i])
                               + norm(up_right_edge_point - new_non_zero_elem[2][i]);
        if (norm_up_max_delta < temp_norm_delta)
        {
            up_max_delta_point = new_non_zero_elem[2][i];
            norm_up_max_delta = temp_norm_delta;
        }
    }

    transformation_points.push_back(down_left_edge_point);
    transformation_points.push_back(up_left_edge_point);
    transformation_points.push_back(up_right_edge_point);
    transformation_points.push_back(
        intersectionLines(down_left_edge_point, down_max_delta_point,
                          up_right_edge_point, up_max_delta_point));

    vector<Point2f> quadrilateral = getQuadrilateral(transformation_points);
    transformation_points = quadrilateral;

    int width = bin_barcode.size().width;
    int height = bin_barcode.size().height;
    for (size_t i = 0; i < transformation_points.size(); i++)
    {
        if ((cvRound(transformation_points[i].x) > width) ||
            (cvRound(transformation_points[i].y) > height)) { return false; }
    }
    return true;
}

Point2f QRDetect::intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2)
{
    Point2f result_square_angle(
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.x - b2.x) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.x - a2.x)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x)),
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.y - b2.y) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.y - a2.y)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x))
                              );
    return result_square_angle;
}

// test function (if true then ------> else <------ )
bool QRDetect::testBypassRoute(vector<Point2f> hull, int start, int finish)
{
    CV_TRACE_FUNCTION();
    int index_hull = start, next_index_hull, hull_size = (int)hull.size();
    double test_length[2] = { 0.0, 0.0 };
    do
    {
        next_index_hull = index_hull + 1;
        if (next_index_hull == hull_size) { next_index_hull = 0; }
        test_length[0] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    index_hull = start;
    do
    {
        next_index_hull = index_hull - 1;
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }
        test_length[1] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    if (test_length[0] < test_length[1]) { return true; } else { return false; }
}

vector<Point2f> QRDetect::getQuadrilateral(vector<Point2f> angle_list)
{
    CV_TRACE_FUNCTION();
    size_t angle_size = angle_list.size();
    uint8_t value, mask_value;
    Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
    Mat fill_bin_barcode = bin_barcode.clone();
    for (size_t i = 0; i < angle_size; i++)
    {
        LineIterator line_iter(bin_barcode, angle_list[ i      % angle_size],
                                            angle_list[(i + 1) % angle_size]);
        for(int j = 0; j < line_iter.count; j++, ++line_iter)
        {
            Point p = line_iter.pos();
            value = bin_barcode.at<uint8_t>(p);
            mask_value = mask.at<uint8_t>(p + Point(1, 1));
            if (value == 0 && mask_value == 0)
            {
                floodFill(fill_bin_barcode, mask, p, 255,
                          0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
            }
        }
    }
    vector<Point> locations;
    Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));

    findNonZero(mask_roi, locations);

    for (size_t i = 0; i < angle_list.size(); i++)
    {
        int x = cvRound(angle_list[i].x);
        int y = cvRound(angle_list[i].y);
        locations.push_back(Point(x, y));
    }

    vector<Point> integer_hull;
    convexHull(locations, integer_hull);
    int hull_size = (int)integer_hull.size();
    vector<Point2f> hull(hull_size);
    for (int i = 0; i < hull_size; i++)
    {
        float x = saturate_cast<float>(integer_hull[i].x);
        float y = saturate_cast<float>(integer_hull[i].y);
        hull[i] = Point2f(x, y);
    }

    const double experimental_area = fabs(contourArea(hull));

    vector<Point2f> result_hull_point(angle_size);
    double min_norm;
    for (size_t i = 0; i < angle_size; i++)
    {
        min_norm = std::numeric_limits<double>::max();
        Point closest_pnt;
        for (int j = 0; j < hull_size; j++)
        {
            double temp_norm = norm(hull[j] - angle_list[i]);
            if (min_norm > temp_norm)
            {
                min_norm = temp_norm;
                closest_pnt = hull[j];
            }
        }
        result_hull_point[i] = closest_pnt;
    }

    int start_line[2] = { 0, 0 }, finish_line[2] = { 0, 0 }, unstable_pnt = 0;
    for (int i = 0; i < hull_size; i++)
    {
        if (result_hull_point[2] == hull[i]) { start_line[0] = i; }
        if (result_hull_point[1] == hull[i]) { finish_line[0] = start_line[1] = i; }
        if (result_hull_point[0] == hull[i]) { finish_line[1] = i; }
        if (result_hull_point[3] == hull[i]) { unstable_pnt = i; }
    }

    int index_hull, extra_index_hull, next_index_hull, extra_next_index_hull;
    Point result_side_begin[4], result_side_end[4];

    bool bypass_orientation = testBypassRoute(hull, start_line[0], finish_line[0]);

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[1]) >
        norm(hull[index_hull] - angle_list[2]) ? angle_list[2] : angle_list[1];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[1], angle_list[2]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[1] - angle_list[2]) * 0.1)
        {
            min_norm = temp_norm;
            result_side_begin[0] = hull[index_hull];
            result_side_end[0]   = hull[next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[0]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[0] = angle_list[1];
        result_side_end[0]   = angle_list[2];
    }

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[1];
    bypass_orientation = testBypassRoute(hull, start_line[1], finish_line[1]);
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[0]) >
        norm(hull[index_hull] - angle_list[1]) ? angle_list[1] : angle_list[0];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[0], angle_list[1]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[0] - angle_list[1]) * 0.05)
        {
            min_norm = temp_norm;
            result_side_begin[1] = hull[index_hull];
            result_side_end[1]   = hull[next_index_hull];
        }

        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[1]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[1] = angle_list[0];
        result_side_end[1]   = angle_list[1];
    }

    bypass_orientation = testBypassRoute(hull, start_line[0], unstable_pnt);
    const bool extra_bypass_orientation = testBypassRoute(hull, finish_line[1], unstable_pnt);

    vector<Point2f> result_angle_list(4), test_result_angle_list(4);
    double min_diff_area = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    const double standart_norm = std::max(
        norm(result_side_begin[0] - result_side_end[0]),
        norm(result_side_begin[1] - result_side_end[1]));
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        if (norm(hull[index_hull] - hull[next_index_hull]) < standart_norm * 0.1)
        { index_hull = next_index_hull; continue; }

        extra_index_hull = finish_line[1];
        do
        {
            if (extra_bypass_orientation) { extra_next_index_hull = extra_index_hull + 1; }
            else { extra_next_index_hull = extra_index_hull - 1; }

            if (extra_next_index_hull == hull_size) { extra_next_index_hull = 0; }
            if (extra_next_index_hull == -1) { extra_next_index_hull = hull_size - 1; }

            if (norm(hull[extra_index_hull] - hull[extra_next_index_hull]) < standart_norm * 0.1)
            { extra_index_hull = extra_next_index_hull; continue; }

            test_result_angle_list[0]
            = intersectionLines(result_side_begin[0], result_side_end[0],
                                result_side_begin[1], result_side_end[1]);
            test_result_angle_list[1]
            = intersectionLines(result_side_begin[1], result_side_end[1],
                                hull[extra_index_hull], hull[extra_next_index_hull]);
            test_result_angle_list[2]
            = intersectionLines(hull[extra_index_hull], hull[extra_next_index_hull],
                                hull[index_hull], hull[next_index_hull]);
            test_result_angle_list[3]
            = intersectionLines(hull[index_hull], hull[next_index_hull],
                                result_side_begin[0], result_side_end[0]);

            const double test_diff_area
                = fabs(fabs(contourArea(test_result_angle_list)) - experimental_area);
            if (min_diff_area > test_diff_area)
            {
                min_diff_area = test_diff_area;
                for (size_t i = 0; i < test_result_angle_list.size(); i++)
                {
                    result_angle_list[i] = test_result_angle_list[i];
                }
            }

            extra_index_hull = extra_next_index_hull;
        }
        while(extra_index_hull != unstable_pnt);

        index_hull = next_index_hull;
    }
    while(index_hull != unstable_pnt);

    // check label points
    if (norm(result_angle_list[0] - angle_list[1]) > 2) { result_angle_list[0] = angle_list[1]; }
    if (norm(result_angle_list[1] - angle_list[0]) > 2) { result_angle_list[1] = angle_list[0]; }
    if (norm(result_angle_list[3] - angle_list[2]) > 2) { result_angle_list[3] = angle_list[2]; }

    // check calculation point
    if (norm(result_angle_list[2] - angle_list[3]) >
       (norm(result_angle_list[0] - result_angle_list[1]) +
        norm(result_angle_list[0] - result_angle_list[3])) * 0.5 )
    { result_angle_list[2] = angle_list[3]; }

    return result_angle_list;
}

//      / | b
//     /  |
//    /   |
//  a/    | c

inline double QRDetect::getCosVectors(Point2f a, Point2f b, Point2f c)
{
    return ((a - b).x * (c - b).x + (a - b).y * (c - b).y) / (norm(a - b) * norm(c - b));
}

struct QRCodeDetector::Impl
{
public:
    Impl() { epsX = 0.2; epsY = 0.1; }
    ~Impl() {}

    double epsX, epsY;
};

QRCodeDetector::QRCodeDetector() : p(new Impl) {}
QRCodeDetector::~QRCodeDetector() {}

void QRCodeDetector::setEpsX(double epsX) { p->epsX = epsX; }
void QRCodeDetector::setEpsY(double epsY) { p->epsY = epsY; }

bool QRCodeDetector::detect(InputArray in, OutputArray points) const
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
        return false;

    QRDetect qrdet;
    qrdet.init(inarr, p->epsX, p->epsY);
    if (!qrdet.localization()) { return false; }
    if (!qrdet.computeTransformationPoints()) { return false; }
    vector<Point2f> pnts2f = qrdet.getTransformationPoints();
    updatePointsResult(points, pnts2f);
    return true;
}

class QRDecode
{
public:
    QRDecode();
    void init(const Mat &src, const vector<Point2f> &points);
    Mat getIntermediateBarcode() { return intermediate; }
    Mat getStraightBarcode() { return straight; }
    size_t getVersion() { return version; }
    std::string getDecodeInformation() { return result_info; }
    bool fullDecodingProcess();

protected:
    bool updatePerspective();
    bool versionDefinition();
    bool samplingForVersion();
    bool decodingProcess();

    decode_error read_format(uint16_t& format,int which);
    decode_error correct_format(uint16_t& format);

    void  unmask_data();
    void  read_data();
    void  read_bit(int x, int y,int& count);
    void  rearrange_blocks();

    decode_error  correct_block(int num , int head,Mat & corrected);

    decode_error decode_payload();
    decode_error decode_numeric(uint8_t * &ptr);

    decode_error decode_byte( uint8_t * &ptr);

    int bits_remaining(const uint8_t *ptr);

    Mat original, no_border_intermediate, intermediate, straight;
    Mat unmasked_data;
    vector<Point2f> original_points;

    uint8_t orignal_data[MAX_PAYLOAD];
    uint8_t rearranged_data[MAX_PAYLOAD];

    Mat final_data;// vector
    uint8_t final[MAX_PAYLOAD];

    int			version;
    int			ecc_level;
    int			mask_type;

    int			data_type;

    uint8_t			payload[ MAX_PAYLOAD];
    int			payload_len;

    uint32_t		eci;

    std::string result_info;
    uint8_t size;
    float test_perspective_size;

};
QRDecode::QRDecode(){
    memset(orignal_data,0,sizeof(uint8_t)*MAX_PAYLOAD);
    memset(rearranged_data,0,sizeof(uint8_t)*MAX_PAYLOAD);
    memset(final,0,sizeof(uint8_t)*MAX_PAYLOAD);
    memset(payload,0,sizeof(uint8_t)*MAX_PAYLOAD);
    payload_len=0;
}

/*
 * params @ format(uint16_t for returning the format bits) which(select from two different place)
 * return @ can be correct or not
 */

decode_error QRDecode::read_format(uint16_t& format , int which){
    /*version<=6*/
    int i;
    uint16_t my_format = 0;

    Mat mat_format(1,FORMAT_LENGTH,CV_8UC1,Scalar(0));

    /*read from the left-bottom and upper-right */
    if (which) {
        /*left-bottom 0-7*/
        for (i = 0; i < 7; i++){
            /*read from pst (code->size - 1 - i,8)*/
            my_format = (my_format << 1) |
                        (straight.ptr<uint8_t>(size - 1 - i)[8]==0);
            mat_format.ptr(0)[i]=(straight.ptr<uint8_t>(size - 1 - i)[8]==0);
        }
        /*upper-right 7-14*/
        for (i = 0; i < 8; i++){
            my_format = (my_format << 1) |
                        (straight.ptr<uint8_t>(8)[size - 8 + i]==0);
            mat_format.ptr(0)[7+i]=(straight.ptr<uint8_t>(8)[size - 8 + i]==0);
        }
    } else{/*read the second format from the upper-left*/
        static const int xs[FORMAT_LENGTH] = {
                8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0
        };
        static const int ys[FORMAT_LENGTH] = {
                0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8
        };
        for (i = FORMAT_LENGTH-1; i >= 0; i--) {
            my_format = (my_format << 1) |
                        (straight.ptr<uint8_t>(ys[i])[xs[i]]==0);

            mat_format.ptr(0)[FORMAT_LENGTH-1-i]=(straight.ptr<uint8_t>(ys[i])[xs[i]]==0);
        }
        std::cout<<std::endl;
    }

    format=my_format;

    return  SUCCESS;
}

/*correct_format:
 *  error correct
 *  params @ uint16_t *f_ret,const Mat& my_f_ret(my format info)
 *  return @
 */
decode_error QRDecode:: correct_format(uint16_t& format)
{
    /*ori: 110101100100011*/
    /*adjust several bits to check the correcting ability*/
    format^=4;format^=8;
    cout<<D2B(format)<<endl;

    /*my method: using BCD ways*/
    int format_index=hamming_detect(format);

    if (format_index==-1)
        return ERROR_FORMAT_ECC;

    format=after_mask_format[format_index]^0x5412;

    return SUCCESS;
}


/*read_bit:
 *  params @ (x,y)
 *  func @ read from (x,y) as the bitpos^th bit of the  bytepos^th codeword
 *  return @
 */
void QRDecode::read_bit(int x, int y, int& count){
    /*judge the reserved area*/
    if (unmasked_data.ptr(y)[x]==INVALID_REGION)
        return ;

    /*the bitpos^th bit of the  bytepos^th codeword*/
    int bytepos = count >> 3;/*equal to count/8 */
    int bitpos  = count & 7 ;/*equal to count%8 */

    int v = (unmasked_data.ptr(y)[x]==0);

    /* first read,first lead*/
    if (v)
        orignal_data[bytepos] |= (0x80 >> bitpos);
    count++;
}

/* exponentiation operator
 * params @  x , power
 * return x^power
 * EXP:
 *     (a^n)^x =a^(x+n)
 * */
uint8_t gf_pow(uint8_t x , int power){
    return gf_exp[(gf_log[x] * power) % 255];
}

uint8_t gf_inverse(uint8_t x){
    return gf_exp[255 - gf_log[x]];
}

/*multiplication in GF
 * params @ x , y
 * return x * y
 * EXP:
 *     a^x * a^y =a^(x+y)
 */
uint8_t gf_mul(const uint8_t &x,const uint8_t& y){
    if (x==0 || y==0)
        return 0;
    return gf_exp[(gf_log[x] + gf_log[y])%255];
}

/*division in GF
 * params @ x , y
 * return x / y
 * EXP:
 *     a^x / a^y =a^(x-y)=a^(x+255-y)
 */
uint8_t gf_div(const uint8_t &x,const uint8_t& y) {
//        CV_Assert (y != 0);
    if (x == 0)
        return 0;
    return gf_exp[(255 - gf_log[y] + gf_log[x]) % 255];
}
/* show_poly
 * params : const Mat& p(polynomial),OUTPUT o(output pattern)
 * return :output string
 * */
std::string show_poly(const Mat& p,OUTPUT o=HEX){
    std::string s;
    for(int i=0;i<p.cols;i++){
        char tmp[10];
        switch (o){
            case HEX:
                sprintf(tmp,"%02X",(int)p.ptr(0)[i]);//
                break;
            case OCT:
                sprintf(tmp,"%d",(int)p.ptr(0)[i]);
                break;
            case ALPHA:
                sprintf(tmp,"%d",gf_log[(int)p.ptr(0)[i]]%255);//%02X
                break;
        }
        s=" "+s;
        s=tmp+s;
    }
    s=s+'\n';
    return  s;
}
/* gf_poly_eval :
 *      evaluate a polynomial at a particular value of x, producing a scalar result
 * params @ poly 15bit format_Info,  uint8_t x(a scalar)
 * return @ result

 * using the Horner's method here:
 * 01 x4 + 0f x3 + 36 x2 + 78 x + 40 = (((01 x + 0f) x + 36) x + 78) x + 40
 * doing this by simple addition and multiplication
 * */
uint8_t gf_poly_eval(const Mat& poly,uint8_t x){
    /*Note the calculation begins at the high times of items,
         * That's to say , start from the large index in Mat
         * */
    int index=poly.cols-1;
    uint8_t y=poly.ptr(0)[index];

    for(int i =index-1;i>=0;i--){
        y = gf_mul(x,y) ^ poly.ptr(0)[i];
    }
    return y;
}
/*
 * func @  multiply a polynomial by a scalar
 * */
Mat gf_poly_scale(const Mat & p,int scalar) {
    int len = p.cols;
    Mat r(1,len,CV_8UC1,Scalar(0));

    for(int i = 0; i < len;i++){
        r.ptr(0)[i] = gf_mul(p.ptr(0)[i], (uint8_t)scalar);
    }
    return r;
}
/*
 * func @  "adds" two polynomials (using exclusive-or, as usual).
 * */
Mat gf_poly_add(const Mat & p,const Mat & q){
    int p_len=p.cols;
    int q_len=q.cols;
    Mat r (1,max(p_len,q_len),CV_8UC1,Scalar(0));

    //int r_len=r.cols;

    for (int i = 0; i< p_len ;i++){
        r.ptr(0)[i] = p.ptr(0)[i];
    }
    for (int i = 0; i< q_len ;i++){
        r.ptr(0)[i] ^= q.ptr(0)[i];//+r_len-q_len
    }
    return r;
}

/* multiplication between two polys
 * params @ poly p , poly q
 * return @ result poly = p * q
 */
/*
       10001001
    *  00101010
 ---------------
      10001001
^   10001001
^ 10001001
 ---------------
  1010001111010*/
Mat gf_poly_mul(const Mat &p,const Mat &q){

    /* multiplication == addition among items*/
    Mat r(1,p.cols+q.cols-1,CV_8UC1,Scalar(0));
    int len_p=p.cols;
    int len_q=q.cols;
    for(int j = 0; j<len_q;j++) {
        if(!q.ptr(0)[j])
            continue;
        //cout<<"round : "<<j<<endl;

        for (int i = 0; i < len_p; i++) {
            if(!p.ptr(0)[i])
                continue;

            r.ptr(0)[i+j] ^= gf_mul(p.ptr(0)[i], q.ptr(0)[j]);
        }

    }
    return r;
}

/*gf_poly_div:
 *   This function is for getting the ECC for the data string ,which is implemented by doing a poly division.
 * params @ const Mat& dividend,const Mat& divisor
 * return @ ECC code/remainder
 *                             12 da df
 *               -----------------------
 *01 0f 36 78 40 ) 12 34 56 00 00 00 00
 *               ^ 12 ee 2b 23 f4
 *              -------------------------
 *                   da 7d 23 f4 00
 *                 ^ da a2 85 79 84
 *                  ---------------------
 *                      df a6 8d 84 00
 *                    ^ df 91 6b fc d9
 *                    -------------------
 *                         37 e6 78 d9
 */
Mat gf_poly_div(const Mat& dividend,const Mat& divisor,const int& ecc_num) {
    /* Note that the processing starts from the item with high number of times,
         * so item [total-i] is processed for the i-th round */
    int times=dividend.cols-(divisor.cols-1);
    int dividend_len=dividend.cols-1;
    int divisor_len=divisor.cols-1;
    /*Mat.ptr(0)[i] stores the coeffient of the x^i*/
    Mat r=dividend.clone();
    for(int i =0;i<times;i++){
        uint8_t coef=r.ptr(0)[dividend_len-i];
        if(coef!=0){
            for (int j = 0; j < divisor.cols; ++j) {
                if(divisor.ptr(0)[divisor_len-j]!=0){
                    r.ptr(0)[dividend_len-i-j]^=gf_mul(divisor.ptr(0)[divisor_len-j], coef);
                }
            }
        }
    }
    Mat ecc=r(Range(0,1),Range(0,ecc_num)).clone();
    return ecc;
}

/*unmask_data
 *func @  unmask the data and make the pixels in the reserved area Scalar(INVALID_REGION)
 */
void QRDecode:: unmask_data(){
    const struct version_info *ver = &version_db[version];
    unmasked_data=straight.clone();
    /*get mask pattern according to the format*/
    int mask_pattren=mask_type;

    for(int i= 0;i<size;i++){
        for(int j= 0;j<size;j++){

            if(unmasked_data.ptr(i)[j]==INVALID_REGION)
                continue;

            /*Finder*/
            if ((i < 9 && j < 9)||/* Finder + format: top left */
                (i + 8 >= size && j < 9)||/* Finder + format: bottom left */
                (i < 9 && j + 8 >= size)||/* Finder + format: top right */
                (i == 6 || j == 6))/* Exclude timing patterns */
            {
                unmasked_data.ptr(i)[j]=INVALID_REGION;
            }
                /*version information*/
            else if (version >= 7) {
                if ((i < 6 && j + 11 >= size)||(i + 11 >= size && j < 6))
                    unmasked_data.ptr(i)[j]=INVALID_REGION;
            }
                /*unmask*/
            else if((mask_pattren==0&&!((i + j) % 2)) ||
                    (mask_pattren==1&&!(i % 2)) ||
                    (mask_pattren==2&&!(j % 3)) ||
                    (mask_pattren==3 && (i + j) % 3 != 0) ||
                    (mask_pattren==4&&!(((i / 2) + (j / 3)) % 2)) ||
                    (mask_pattren==5&&!((i * j) % 2 + (i * j) % 3))||
                    (mask_pattren==6&&!(((i * j) % 2 + (i * j) % 3) % 2))||
                    ((mask_pattren==7 && ((i * j) % 3 + (i + j) % 2) % 2 != 0))
                    ){
                unmasked_data.ptr(i)[j]^=255;
            }
        }
    }

    /* Exclude alignment patterns */
    for (int a = 0; a < MAX_ALIGNMENT && ver->apat[a]; a++) {
        for (int p = a; p < MAX_ALIGNMENT && ver->apat[p]; p++) {
            int x=ver->apat[a];
            int y=ver->apat[p];
            /*the alignment patterns MUST NOT overlap the finder patterns or separators*/
            if(unmasked_data.ptr(x)[y]==INVALID_REGION)
                continue;
            for(int i=-2;i<=2;i++)
                for(int j=-2;j<=2;j++)
                    unmasked_data.ptr(x+i)[y+j]=INVALID_REGION;
        }
    }

}

/*read_data
 * func @ read data from the image into codewords in a zig-zag way
 * */
void QRDecode::read_data(){
    int y = size - 1;
    int x = size - 1;
    int dir = -1;
    int count = 0;
    while (x > 0) {
        if (x == 6)
            x--;
        /*read*/
        read_bit( x,  y, count);
        read_bit( x-1,  y, count);

        y += dir;
        /*change direction when meets border*/
        if (y < 0 || y >= size) {
            dir = -dir;
            x -= 2;
            y += dir;
        }
    }
}

/*block_syndromes
 * params @ const Mat & block(current block),
 *          int synd_num(the num of the syndromes),
 *          uint8_t *synd(syndromes for output)
 * func   @ calculate the syndromes for each block
 * */
int block_syndromes(const Mat & block, int synd_num,vector <uint8_t>& synd){
    int nonzero = 0;
    /*the original method*/
    cout<<"\n@s@ "<<endl;
    for (int i = 0; i < synd_num; i++) {
        /*get the syndromes by repalcing the x with pow(2,i) and evaluating the results of the equations*/
        uint8_t tmp =gf_poly_eval(block, gf_pow(2,i));
        /*print for debug*/
        cout<<(int)tmp<<" ";
        if (tmp)
            nonzero = 1;
        synd.push_back(tmp);
    }
    cout<<endl;
    return nonzero;
}


/*find_error_locator
 * params @ synd(the syndromes of current block),
 *          errors_len(the number of the errors),
 * func   @ using berlekamp_massey algorithm to calculate the error_locator
 * return @ find_error_locator
 * */
Mat find_error_locator(const vector<uint8_t>&synd,int & errors_len){
    /*initialize two arrays b and c ,to be zeros , expcet b0<- 1 c0<- 1*/
    int synd_num =(int)synd.size();
    /*err_loc & Sigma*/
    Mat C(1,synd_num,CV_8UC1,Scalar(0));
    /*old_loc */
    Mat B(1,synd_num,CV_8UC1,Scalar(0));//a copy of the last C

    B.ptr(0)[0]=1;
    C.ptr(0)[0]=1;

    uint8_t b=1;//a copy of the last discrepancy delta
    int L = 0;//the current number of assumed errors
    int m = 1;//the number of iterations

    for(int i = 0; i < synd_num ;i++){
        uint8_t delta = synd[i];
        /*cal discrepancy =Sn+ C1*S(n-1) + ... + CL*S(n-L)*/
        for(int j = 1;j<=L;j++){
            delta ^= gf_mul(C.ptr(0)[j], synd[i - j]);
        }
        /*shift = x^m*/
        Mat shift(1,synd_num,CV_8UC1,Scalar(0));
        shift.ptr(0)[m]=1;
        /*scale_coeffi = d/b */
        Mat scale_coeffi = gf_poly_scale(shift,gf_mul(delta,gf_inverse(b)));

        /*if delta == 0 c is the polynomial */
        if(delta == 0){
            /*assumes that C(x) and L are correct for the moment, increments m, and continues*/
            m++;

        }
            /*If delta !=0 , adjust C(x) so that a recalculation of d would be zero*/

        else if(2 * L <= i){
            /*L is updated and algorithm will update B(x), b, increase L, and reset m = 1*/
            Mat t=C.clone();
            /*C(t)=C(t)-(d/b)*x^m*B(t)    //t stands for the iteration times
                 *    =C(t)-(d/b)*x^m*C(t-1)
                 *delta = Sn+ C1*S(n-1) + ... -(d/b)(Sj+ B1*S(j-1) + ...)
                 */
            C=gf_poly_add(C,gf_poly_mul(B,scale_coeffi));

            B = t.clone();
            b=delta;

            L = i + 1 - L;
            m = 1;
        }

        else{
            /*If L equals the actual number of errors, then during the iteration process,
                 * the discrepancies will become zero before n becomes greater than or equal to 2L.*/

            C = gf_poly_add(C, gf_poly_mul(B,scale_coeffi));
            m++;
        }

    }
    cout<<"len : "<< L<<endl;
    errors_len=L;
    /*L is the length of the minimal LFSR for the stream*/
    cout<<"@@sigma@@"<<endl;
    cout<<C<<endl;
    return C;
}

/*
     * params @ sigma(error locator) , errors_len(length of sigma -1) , msg_len(length of the block)
     * return @ error index(the index of the error in current block)
     * func   @ use Chien's search to calculate the the roots of error locator poly
     *          and to calculate the index of the error in current block
     * */
vector<int > find_errors(const Mat& sigma,const int &errors_len,const int & msg_len){
    vector <int> error_index;
    /*optimize to just check the interesting symbols*/
    for(int i = 0; i < msg_len ; i ++){
        int index=msg_len-i-1;
        /* if a^(n-i) is the error postion ,then a^-(n-i) is the root of the poly Sigma
             * use Chien's search to evaluate the polynomial such that each evaluation only takes constant time
             */
        if(gf_poly_eval(sigma,gf_inverse(gf_pow(2,index)))==0){
            error_index.push_back(index);
        }
    }
    /*print out for debugging*/

    CV_Assert((int)error_index.size()==errors_len);

    return error_index;

}
/*
     * params @ msg_in(the orignial blcok) , synd(syndromes) ,
     *          e_loc_poly(error locator poly) , error_index(the index of the error)
     * return @ the corrected block
     * func   @ use Forney algorithm to correct the errors
     *  ps :
     *  Error location polynomial (short for A(x)) = 1 + sigma{lambda(i) * x(i)}
     *  A(x)'=sigma{i * lambda(i) * x(i-1)}
     * */
Mat error_correct(const Mat & msg_in ,const vector<uint8_t>&synd,const Mat & e_loc_poly,const vector<int> &error_index){

    int border = (int)synd.size();
    Mat msg_out = msg_in.clone() ;
    int err_len= (int)error_index.size();

    Mat syndrome(1,border,CV_8UC1,Scalar(0));
    /*change syndrom to mat from calculation*/

    for(int i = 1 ; i < border ; i++){
        syndrome.ptr(0)[i]=synd[i];//border-1-
    }


    /*First calculate the error evaluator polynomial*/
    Mat Omega= gf_poly_mul(syndrome,e_loc_poly);

    /*get rid of the first zero ! */
    Omega = Omega(Range(0,1),Range(1,border));

    /*Second use Forney algorithm to compute the magnitudes*/

    /* 1. calculate formal derivative as the denominator */
    Mat err_location_poly_derivative(1,e_loc_poly.cols,CV_8UC1,Scalar(0));

    /*The operator · represents ordinary multiplication (repeated addition in the finite field)!!!
         * that's why the even items are always zero!*/
    for(int i = 1;i < err_len; i++){
        uint8_t tmp = e_loc_poly.ptr(0)[i];
        err_location_poly_derivative.ptr(0)[i-1]=tmp;
        for(int j = 1; j < i ; j++)
            err_location_poly_derivative.ptr(0)[i-1]^=tmp;
    }
    /* 2. calculate Omega as the numerator*/
    for (int i = 0; i < err_len; i++) {
        //int root = total_len-error_index[i]-1;
        uint8_t xinv = gf_inverse(gf_pow(2, error_index[i]));

        uint8_t denominator = gf_poly_eval(err_location_poly_derivative,xinv);

        uint8_t numerator = gf_poly_eval(Omega, xinv);

        /*divded them to get the magnitude*/
        uint8_t error_magnitude = gf_div(numerator,denominator);

        msg_out.ptr(0)[error_index[i]]^=error_magnitude;

    }
    return msg_out;
}

/* correct_block
 * params @ num(the number of the current block)  head(the beginning index of NUM^th block)
 * func   @ correct NUM^th block
 * */
decode_error  QRDecode::correct_block(int num , int head,Mat & corrected){
    const version_info *ver =&version_db[version];
    const  block_params *cur_ecc = &ver->ecc[ecc_code2level(ecc_level)];

    int cur_length=0;

    int ecc_num=cur_ecc->ecc_codewords;

    if(num<cur_ecc->num_blocks_in_G1){
        cur_length=cur_ecc->data_codewords_in_G1+ecc_num;
    }
    else{
        cur_length=cur_ecc->data_codewords_in_G2+ecc_num;
    }
    vector<uint8_t>synd;

    /*get the block for block_syndromes*/
    Mat cur_block(1,cur_length,CV_8UC1,Scalar(0));
    for(int i = 0 ;i<cur_length ; i++) {
        cur_block.ptr(0)[cur_length-1-i]=rearranged_data[head+i];
    }

    corrected=cur_block.clone();

    if (!block_syndromes(cur_block,ecc_num,synd))
        return SUCCESS;

    int errors_len=0;
    Mat sigma=find_error_locator(synd,errors_len);

    vector <int> error_index = find_errors(sigma,errors_len,cur_length);

    Mat corrected_block = error_correct(cur_block ,synd, sigma, error_index);

    /*check once again*/
    if (block_syndromes(corrected_block,ecc_num,synd))
        return ERROR_DATA_ECC;

    corrected=corrected_block.clone();

    return SUCCESS;
}

/*
 * params @
 * func   @ rearrange the interleaved blocks for later codewode correction
 * */
void QRDecode::rearrange_blocks(){
    const version_info *ver =&version_db[version];
    const  block_params *cur_ecc = &ver->ecc[ecc_code2level(ecc_level)];

    //{  18,	2,	15,	2,	16},
    int index=0;
    int count=0;
    int my_count = 0;
    int offset=cur_ecc->num_blocks_in_G1+cur_ecc->num_blocks_in_G2;

    /*the beginning of ecc*/
    int offset_ecc= cur_ecc->data_codewords_in_G1*cur_ecc->num_blocks_in_G1
                    +
                    cur_ecc->data_codewords_in_G2*cur_ecc->num_blocks_in_G2;

    final_data=Mat(1,offset_ecc*8,CV_8UC1,Scalar(0)).clone();

    /*total num of blocks*/
    int total_blocks=cur_ecc->num_blocks_in_G1+cur_ecc->num_blocks_in_G2;

    /*the offset for one more col in G2*/
    int offset_one_more=total_blocks*cur_ecc->data_codewords_in_G1;


    int cur_block_head=0;
    /*get block in group1*/
    for(int i =0;i<total_blocks;i++){

        /*get the data codeword*/
        for(int j = 0;j <cur_ecc->data_codewords_in_G1;j++){
            rearranged_data[index]=orignal_data[i+j*offset];

            if(rearranged_data[index]==71)
                rearranged_data[index]=7;
            if(rearranged_data[index]==3)
                rearranged_data[index]=30;
            index++;

        }
        /*one more  col in G2*/
        if(i>=cur_ecc->num_blocks_in_G1){
            rearranged_data[index++]=orignal_data[offset_one_more+i-cur_ecc->num_blocks_in_G1];
        }

        /*get the ecc codeword*/
        for(int j = 0;j <cur_ecc->ecc_codewords;j++){
            rearranged_data[index++]=orignal_data[offset_ecc+i+j*offset];
        }

        Mat  corrected;
        decode_error err= correct_block(i,cur_block_head,corrected);

        int border = (i>=cur_ecc->num_blocks_in_G1)?cur_ecc->data_codewords_in_G2:cur_ecc->data_codewords_in_G1;

        int total =border +cur_ecc->ecc_codewords;

        for(int j = 0 ;j < border ; j++){
            final[count++]=corrected.ptr(0)[total-1-j];
        }

        std::string s =" " ;
        for(int j = 0 ; j < border*CODEWORD_LEN; j++){
            int cur_word=j>>3;
            int cur_bit =CODEWORD_LEN-1-(j&7);
            final_data.ptr(0)[my_count++]=(corrected.ptr(0)[total-1-cur_word]>>(cur_bit)) & (1);
        }
        cur_block_head=index;

        if(err)
            return;
    }
}


void QRDecode::init(const Mat &src, const vector<Point2f> &points)
{
    CV_TRACE_FUNCTION();
    vector<Point2f> bbox = points;
    original = src.clone();
    intermediate = Mat::zeros(original.size(), CV_8UC1);
    original_points = bbox;
    version = 0;
    size = 0;
    test_perspective_size = 251;
    result_info = "";
}

bool QRDecode::updatePerspective()
{
    CV_TRACE_FUNCTION();
    const Point2f centerPt = QRDetect::intersectionLines(original_points[0], original_points[2],
                                                         original_points[1], original_points[3]);
    if (cvIsNaN(centerPt.x) || cvIsNaN(centerPt.y))
        return false;

    const Size temporary_size(cvRound(test_perspective_size), cvRound(test_perspective_size));

    vector<Point2f> perspective_points;
    perspective_points.push_back(Point2f(0.f, 0.f));
    perspective_points.push_back(Point2f(test_perspective_size, 0.f));

    perspective_points.push_back(Point2f(test_perspective_size, test_perspective_size));
    perspective_points.push_back(Point2f(0.f, test_perspective_size));

    perspective_points.push_back(Point2f(test_perspective_size * 0.5f, test_perspective_size * 0.5f));

    vector<Point2f> pts = original_points;
    pts.push_back(centerPt);

    Mat H = findHomography(pts, perspective_points);
    Mat bin_original;
    adaptiveThreshold(original, bin_original, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    Mat temp_intermediate;
    warpPerspective(bin_original, temp_intermediate, H, temporary_size, INTER_NEAREST);
    no_border_intermediate = temp_intermediate(Range(1, temp_intermediate.rows), Range(1, temp_intermediate.cols));

    const int border = cvRound(0.1 * test_perspective_size);
    const int borderType = BORDER_CONSTANT;
    copyMakeBorder(no_border_intermediate, intermediate, border, border, border, border, borderType, Scalar(255));
    return true;
}

inline Point computeOffset(const vector<Point>& v)
{
    // compute the width/height of convex hull
    Rect areaBox = boundingRect(v);

    // compute the good offset
    // the box is consisted by 7 steps
    // to pick the middle of the stripe, it needs to be 1/14 of the size
    const int cStep = 7 * 2;
    Point offset = Point(areaBox.width, areaBox.height);
    offset /= cStep;
    return offset;
}

bool QRDecode::versionDefinition()
{
    CV_TRACE_FUNCTION();
    LineIterator line_iter(intermediate, Point2f(0, 0), Point2f(test_perspective_size, test_perspective_size));
    Point black_point = Point(0, 0);
    for(int j = 0; j < line_iter.count; j++, ++line_iter)
    {
        const uint8_t value = intermediate.at<uint8_t>(line_iter.pos());
        if (value == 0)
        {
            black_point = line_iter.pos();
            break;
        }
    }

    Mat mask = Mat::zeros(intermediate.rows + 2, intermediate.cols + 2, CV_8UC1);
    floodFill(intermediate, mask, black_point, 255, 0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);

    vector<Point> locations, non_zero_elem;
    Mat mask_roi = mask(Range(1, intermediate.rows - 1), Range(1, intermediate.cols - 1));
    findNonZero(mask_roi, non_zero_elem);
    convexHull(non_zero_elem, locations);
    Point offset = computeOffset(locations);

    Point temp_remote = locations[0], remote_point;
    const Point delta_diff = offset;
    for (size_t i = 0; i < locations.size(); i++)
    {
        if (norm(black_point - temp_remote) <= norm(black_point - locations[i]))
        {
            const uint8_t value = intermediate.at<uint8_t>(temp_remote - delta_diff);
            temp_remote = locations[i];
            if (value == 0) { remote_point = temp_remote - delta_diff; }
            else { remote_point = temp_remote - (delta_diff / 2); }
        }
    }

    size_t transition_x = 0 , transition_y = 0;

    uint8_t future_pixel = 255;
    const uint8_t *intermediate_row = intermediate.ptr<uint8_t>(remote_point.y);
    for(int i = remote_point.x; i < intermediate.cols; i++)
    {
        if (intermediate_row[i] == future_pixel)
        {
            future_pixel = static_cast<uint8_t>(~future_pixel);
            transition_x++;
        }
    }

    future_pixel = 255;
    for(int j = remote_point.y; j < intermediate.rows; j++)
    {
        const uint8_t value = intermediate.at<uint8_t>(Point(j, remote_point.x));
        if (value == future_pixel)
        {
            future_pixel = static_cast<uint8_t>(~future_pixel);
            transition_y++;
        }
    }
    version = saturate_cast<uint8_t>((std::min(transition_x, transition_y) - 1) * 0.25 - 1);
    if ( !(  0 < version && version <= 40 ) ) { return false; }
    size = (uint8_t)(21 + (version - 1) * 4);
    return true;
}

bool QRDecode::samplingForVersion()
{
    CV_TRACE_FUNCTION();
    const double multiplyingFactor = (version < 3)  ? 1 :
                                     (version == 3) ? 1.5 :
                                     version * (5 + version - 4);
    const Size newFactorSize(
                  cvRound(no_border_intermediate.size().width  * multiplyingFactor),
                  cvRound(no_border_intermediate.size().height * multiplyingFactor));
    Mat postIntermediate(newFactorSize, CV_8UC1);
    resize(no_border_intermediate, postIntermediate, newFactorSize, 0, 0, INTER_AREA);

    const int delta_rows = cvRound((postIntermediate.rows * 1.0) / size);
    const int delta_cols = cvRound((postIntermediate.cols * 1.0) / size);

    vector<double> listFrequencyElem;
    for (int r = 0; r < postIntermediate.rows; r += delta_rows)
    {
        for (int c = 0; c < postIntermediate.cols; c += delta_cols)
        {
            Mat tile = postIntermediate(
                           Range(r, min(r + delta_rows, postIntermediate.rows)),
                           Range(c, min(c + delta_cols, postIntermediate.cols)));
            const double frequencyElem = (countNonZero(tile) * 1.0) / tile.total();
            listFrequencyElem.push_back(frequencyElem);
        }
    }

    double dispersionEFE = std::numeric_limits<double>::max();
    double experimentalFrequencyElem = 0;
    for (double expVal = 0; expVal < 1; expVal+=0.001)
    {
        double testDispersionEFE = 0.0;
        for (size_t i = 0; i < listFrequencyElem.size(); i++)
        {
            testDispersionEFE += (listFrequencyElem[i] - expVal) *
                                 (listFrequencyElem[i] - expVal);
        }
        testDispersionEFE /= (listFrequencyElem.size() - 1);
        if (dispersionEFE > testDispersionEFE)
        {
            dispersionEFE = testDispersionEFE;
            experimentalFrequencyElem = expVal;
        }
    }

    straight = Mat(Size(size, size), CV_8UC1, Scalar(0));
    for (int r = 0; r < size * size; r++)
    {
        int i   = r / straight.cols;
        int j   = r % straight.cols;
        straight.ptr<uint8_t>(i)[j] = (listFrequencyElem[r] < experimentalFrequencyElem) ? 0 : 255;
    }
    return true;
}
/* get_bits
 * params @ bits(the number of bits you need) ptr(the starting position)
 * func   @ from the postion $PTR to get $BITS bits
 * */
int get_bits(const int& bits,uint8_t * & ptr){
    int result=0;
    for(int i =0 ;i<bits;i++){

        result=result<<1;
        result+=*(ptr+i);//data.ptr(0)[ptr+i];
    }
    ptr+=bits;
    return result;
}

/* bits_remaining
 * params @ ptr(current bit postion)
 * func   @ calculate the remaining number of bits
 * */
int QRDecode::bits_remaining(const uint8_t *ptr)
{
    uint8_t * tail = &final_data.ptr(0)[final_data.cols-1];
    return (int)(tail - ptr);
}
/* decode_numeric
 * params @ ptr(current bit postion)
 * func   @ decode the numerical mode
 * */
decode_error QRDecode::decode_numeric(uint8_t * &ptr){
    int count = 0;

    /*check version to update the bit counter*/
    int bits = 10;
    if(version>=27)
        bits=14;
    else if(version>=10)
        bits=12;

    count = get_bits(bits,ptr);

    if (payload_len + count + 1 > MAX_PAYLOAD){
        return ERROR_DATA_OVERFLOW;
    }
    /*divided 3 numerical char into a 10bit group*/
    while (count >= 3) {
        int num = get_bits(10,ptr);
        payload[payload_len++]= uint8_t(num/100+'0');
        payload[payload_len++]= uint8_t((num%100)/10+'0');
        payload[payload_len++]= uint8_t(num%10+'0');
        count -= 3;
    }
    /*the final group*/
    if(count == 2){
        /*7 bit group*/
        int num = get_bits(7,ptr);
        payload[payload_len++] = uint8_t((num%100)/10+'0');
        payload[payload_len++] =  uint8_t(num%10+'0');
    }
    else if (count == 1){
        /*4 bit group*/
        int num = get_bits(4,ptr);
        payload[payload_len++] = uint8_t(num%10+'0');
    }

    return SUCCESS;

}
/* decode_byte
 * params @ ptr(current bit postion)
 * func   @ decode the byte mode
 * */
decode_error QRDecode::decode_byte(uint8_t * &ptr){
    int bits = 8;
    int count = 0;
    /*check version to update the bit counter*/
    if(version>9)
        bits=16;

    count = get_bits(bits,ptr);

    if (payload_len + count + 1 > MAX_PAYLOAD){
        return ERROR_DATA_OVERFLOW;
    }

    if (bits_remaining(ptr) < count * 8){
        return ERROR_DATA_UNDERFLOW;
    }

    for (int i = 0; i < count; i++){
        int tmp =get_bits(8,ptr);
        payload[payload_len++]= uint8_t(tmp);
    }

    return SUCCESS;
}
/* decode_payload
 * params @
 * func   @ decode the data according to its corresponding mode
 * */
decode_error QRDecode::decode_payload(){
    decode_error err = SUCCESS;
    uint8_t * ptr = &final_data.ptr(0)[0];
   /*test for output*/
   // output_final_data(final_data);

    while(bits_remaining(ptr)>=4){
        int mode=get_bits(4,ptr);
        /*select the corresponding decode mode */
        switch (mode){
            case QR_MODE_NUL:
                ptr = &final_data.ptr(0)[final_data.cols-1];
                break;
            case QR_MODE_NUM:
                err = decode_numeric(ptr);
                break;
            case QR_MODE_BYTE:
                err = decode_byte(ptr);
                break;


        }
    }
    if(err)
        return ERROR_UNKNOWN_DATA_TYPE;
    return SUCCESS;
}

bool QRDecode::decodingProcess()
{
#ifdef HAVE_QUIRC
    if (straight.empty()) { return false; }

    quirc_code qr_code;
    memset(&qr_code, 0, sizeof(qr_code));

    qr_code.size = straight.size().width;
    for (int x = 0; x < qr_code.size; x++)
    {
        for (int y = 0; y < qr_code.size; y++)
        {
            int position = y * qr_code.size + x;
            qr_code.cell_bitmap[position >> 3]
                |= straight.ptr<uint8_t>(y)[x] ? 0 : (1 << (position & 7));
        }
    }

    quirc_data qr_code_data;
    quirc_decode_error_t errorCode = quirc_decode(&qr_code, &qr_code_data);
    if (errorCode != 0) { return false; }

    for (int i = 0; i < qr_code_data.payload_len; i++)
    {
        result_info += qr_code_data.payload[i];
    }
    return true;
#else
    decode_error err;
    uint16_t my_format=0;
    if (straight.empty()) { return false; }
    size=straight.size().width;


    if ((size - 17) % 4)
        return ERROR_INVALID_GRID_SIZE;

    /*estimated version*/
    version = (size - 17) / 4;

    if (version < 1 ||
        version > MAX_VERSION)
        return ERROR_INVALID_VERSION;

    /* Read format information -- try both locations */
    err = read_format(my_format,0);
    if (err)
        err = read_format(my_format,1);
    if (err)
        return err;

    /*ECC correction*/
    err = correct_format(my_format);
    if (err)
        return err;

    /*EC level （1-2）+Mask(3-5) + EC for this string( 6-15) */
    /*get rid of the ecc_code*/
    u_int8_t fdata = my_format >> 10;
    ecc_level = fdata >> 3;
    mask_type = fdata & 7;

    unmask_data();

    read_data();

    rearrange_blocks();

    if (err != 0) { return false; }

    return true;
#endif
}

bool QRDecode::fullDecodingProcess()
{
#ifdef HAVE_QUIRC
    if (!updatePerspective())  { return false; }
    if (!versionDefinition())  { return false; }
    if (!samplingForVersion()) { return false; }
    if (!decodingProcess())    { return false; }
    return true;
#else
    if (!updatePerspective())  { return false; }
    if (!versionDefinition())  { return false; }
    if (!samplingForVersion()) { return false; }
    if (!decodingProcess())    { return false; }
    return true;
    std::cout << "Library QUIRC is not linked. No decoding is performed. Take it to the OpenCV repository." << std::endl;
    return false;
#endif
}

bool decodeQRCode(InputArray in, InputArray points, std::string &decoded_info, OutputArray straight_qrcode)
{
    QRCodeDetector qrcode;
    decoded_info = qrcode.decode(in, points, straight_qrcode);
    return !decoded_info.empty();
}

cv::String QRCodeDetector::decode(InputArray in, InputArray points,
                                  OutputArray straight_qrcode)
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
        return std::string();

    vector<Point2f> src_points;
    points.copyTo(src_points);
    CV_Assert(src_points.size() == 4);
    CV_CheckGT(contourArea(src_points), 0.0, "Invalid QR code source points");

    QRDecode qrdec;
    qrdec.init(inarr, src_points);
    bool ok = qrdec.fullDecodingProcess();

    std::string decoded_info = qrdec.getDecodeInformation();

    if (ok && straight_qrcode.needed())
    {
        qrdec.getStraightBarcode().convertTo(straight_qrcode,
                                             straight_qrcode.fixedType() ?
                                             straight_qrcode.type() : CV_32FC2);
    }

    return ok ? decoded_info : std::string();
}

cv::String QRCodeDetector::detectAndDecode(InputArray in,
                                           OutputArray points_,
                                           OutputArray straight_qrcode)
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
    {
        points_.release();
        return std::string();
    }

    vector<Point2f> points;
    bool ok = detect(inarr, points);
    if (!ok)
    {
        points_.release();
        return std::string();
    }
    updatePointsResult(points_, points);
    std::string decoded_info = decode(inarr, points, straight_qrcode);
    return decoded_info;
}

bool detectQRCode(InputArray in, vector<Point> &points, double eps_x, double eps_y)
{
    QRCodeDetector qrdetector;
    qrdetector.setEpsX(eps_x);
    qrdetector.setEpsY(eps_y);

    return qrdetector.detect(in, points);
}

class QRDetectMulti : public QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints(const size_t cur_ind);
    vector< vector < Point2f > > getTransformationPoints() { return transformation_points;}

protected:
    int findNumberLocalizationPoints(vector<Point2f>& tmp_localization_points);
    void findQRCodeContours(vector<Point2f>& tmp_localization_points, vector< vector< Point2f > >& true_points_group, const int& num_qrcodes);
    bool checkSets(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& true_points_group_copy,
                   vector<Point2f>& tmp_localization_points);
    void deleteUsedPoints(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& loc,
                          vector<Point2f>& tmp_localization_points);
    void fixationPoints(vector<Point2f> &local_point);
    bool checkPoints(vector<Point2f> quadrangle_points);
    bool checkPointsInsideQuadrangle(const vector<Point2f>& quadrangle_points);
    bool checkPointsInsideTriangle(const vector<Point2f>& triangle_points);

    Mat bin_barcode_fullsize, bin_barcode_temp;
    vector<Point2f> not_resized_loc_points;
    vector<Point2f> resized_loc_points;
    vector< vector< Point2f > > localization_points, transformation_points;
    struct compareDistanse_y
    {
        bool operator()(const Point2f& a, const Point2f& b) const
        {
            return a.y < b.y;
        }
    };
    struct compareSquare
    {
        const vector<Point2f>& points;
        compareSquare(const vector<Point2f>& points_) : points(points_) {}
        bool operator()(const Vec3i& a, const Vec3i& b) const;
    };
    Mat original;
    class ParallelSearch : public ParallelLoopBody
    {
    public:
        ParallelSearch(vector< vector< Point2f > >& true_points_group_,
                vector< vector< Point2f > >& loc_, int iter_, vector<int>& end_,
                vector< vector< Vec3i > >& all_points_,
                QRDetectMulti& cl_)
        :
            true_points_group(true_points_group_),
            loc(loc_),
            iter(iter_),
            end(end_),
            all_points(all_points_),
            cl(cl_)
        {
        }
        void operator()(const Range& range) const CV_OVERRIDE;
        vector< vector< Point2f > >& true_points_group;
        vector< vector< Point2f > >& loc;
        int iter;
        vector<int>& end;
        vector< vector< Vec3i > >& all_points;
        QRDetectMulti& cl;
    };
};

void QRDetectMulti::ParallelSearch::operator()(const Range& range) const
{
    for (int s = range.start; s < range.end; s++)
    {
        bool flag = false;
        for (int r = iter; r < end[s]; r++)
        {
            if (flag)
                break;

            size_t x = iter + s;
            size_t k = r - iter;
            vector<Point2f> triangle;

            for (int l = 0; l < 3; l++)
            {
                triangle.push_back(true_points_group[s][all_points[s][k][l]]);
            }

            if (cl.checkPointsInsideTriangle(triangle))
            {
                bool flag_for_break = false;
                cl.fixationPoints(triangle);
                if (triangle.size() == 3)
                {
                    cl.localization_points[x] = triangle;
                    if (cl.purpose == cl.SHRINKING)
                    {

                        for (size_t j = 0; j < 3; j++)
                        {
                            cl.localization_points[x][j] *= cl.coeff_expansion;
                        }
                    }
                    else if (cl.purpose == cl.ZOOMING)
                    {
                        for (size_t j = 0; j < 3; j++)
                        {
                            cl.localization_points[x][j] /= cl.coeff_expansion;
                        }
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        for (size_t j = i + 1; j < 3; j++)
                        {
                            if (norm(cl.localization_points[x][i] - cl.localization_points[x][j]) < 10)
                            {
                                cl.localization_points[x].clear();
                                flag_for_break = true;
                                break;
                            }
                        }
                        if (flag_for_break)
                            break;
                    }
                    if ((!flag_for_break)
                            && (cl.localization_points[x].size() == 3)
                            && (cl.computeTransformationPoints(x))
                            && (cl.checkPointsInsideQuadrangle(cl.transformation_points[x]))
                            && (cl.checkPoints(cl.transformation_points[x])))
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            loc[s][all_points[s][k][l]].x = -1;
                        }

                        flag = true;
                        break;
                    }
                }
                if (flag)
                {
                    break;
                }
                else
                {
                    cl.transformation_points[x].clear();
                    cl.localization_points[x].clear();
                }
            }
        }
    }
}

void QRDetectMulti::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();

    CV_Assert(!src.empty());
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        purpose = ZOOMING;
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        const int width  = cvRound(src.size().width  / coeff_expansion);
        const int height = cvRound(src.size().height  / coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_AREA);
    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
        barcode = src.clone();
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;
    adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    adaptiveThreshold(src, bin_barcode_fullsize, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
}

void QRDetectMulti::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();

    Point2f v0(local_point[1] - local_point[2]);
    Point2f v1(local_point[0] - local_point[2]);
    Point2f v2(local_point[1] - local_point[0]);

    double cos_angles[3], norm_triangl[3];
    norm_triangl[0] = norm(v0);
    norm_triangl[1] = norm(v1);
    norm_triangl[2] = norm(v2);

    cos_angles[0] = v2.dot(-v1) / (norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = v2.dot(v0) / (norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = v1.dot(v0) / (norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
        local_point.clear();
        return;
    }

    size_t i_min_cos =
            (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
                    (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]);
        const Point2f left_point(local_point[left_index]);
        const Point2f right_point(local_point[right_index]);
        const Point2f central_point(intersectionLines(
                current_point,
                Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                        static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                Point2f(0, static_cast<float>(bin_barcode_temp.rows - 1)),
                Point2f(static_cast<float>(bin_barcode_temp.cols - 1),
                        static_cast<float>(bin_barcode_temp.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            LineIterator& li = list_line_iter[k];
            uint8_t future_pixel = 255, count_index = 0;
            for (int j = 0; j < li.count; j++, ++li)
            {
                Point p = li.pos();
                if (p.x >= bin_barcode_temp.cols ||
                    p.y >= bin_barcode_temp.rows)
                {
                    break;
                }

                const uint8_t value = bin_barcode_temp.at<uint8_t>(p);
                if (value == future_pixel)
                {
                    future_pixel = static_cast<uint8_t>(~future_pixel);
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(p);
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos)
    {
        std::swap(local_point[0], local_point[index_max]);
    }
    else
    {
        local_point.clear();
        return;
    }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if (determinant(m) > 0)
    {
        std::swap(local_point[1], local_point[2]);
    }
}

class BWCounter
{
    size_t white;
    size_t black;
public:
    BWCounter(size_t b = 0, size_t w = 0) : white(w), black(b) {}
    BWCounter& operator+=(const BWCounter& other) { black += other.black; white += other.white; return *this; }
    void count1(uchar pixel) { if (pixel == 255) white++; else if (pixel == 0) black++; }
    double getBWFraction() const { return white == 0 ? std::numeric_limits<double>::infinity() : double(black) / double(white); }
    static BWCounter checkOnePair(const Point2f& tl, const Point2f& tr, const Point2f& bl, const Point2f& br, const Mat& img)
    {
        BWCounter res;
        LineIterator li1(img, tl, tr), li2(img, bl, br);
        for (int i = 0; i < li1.count && i < li2.count; i++, li1++, li2++)
        {
            LineIterator it(img, li1.pos(), li2.pos());
            for (int r = 0; r < it.count; r++, it++)
                res.count1(img.at<uchar>(it.pos()));
        }
        return res;
    }
};

bool QRDetectMulti::checkPoints(vector<Point2f> quadrangle)
{
    if (quadrangle.size() != 4)
        return false;
    std::sort(quadrangle.begin(), quadrangle.end(), compareDistanse_y());
    BWCounter s;
    s += BWCounter::checkOnePair(quadrangle[1], quadrangle[0], quadrangle[2], quadrangle[0], bin_barcode);
    s += BWCounter::checkOnePair(quadrangle[1], quadrangle[3], quadrangle[2], quadrangle[3], bin_barcode);
    const double frac = s.getBWFraction();
    return frac > 0.76 && frac < 1.24;
}

bool QRDetectMulti::checkPointsInsideQuadrangle(const vector<Point2f>& quadrangle_points)
{
    if (quadrangle_points.size() != 4)
        return false;

    int count = 0;
    for (size_t i = 0; i < not_resized_loc_points.size(); i++)
    {
        if (pointPolygonTest(quadrangle_points, not_resized_loc_points[i], true) > 0)
        {
            count++;
        }
    }
    if (count == 3)
        return true;
    else
        return false;
}

bool QRDetectMulti::checkPointsInsideTriangle(const vector<Point2f>& triangle_points)
{
    if (triangle_points.size() != 3)
        return false;
    double eps = 3;
    for (size_t i = 0; i < resized_loc_points.size(); i++)
    {
        if (pointPolygonTest( triangle_points, resized_loc_points[i], true ) > 0)
        {
            if ((abs(resized_loc_points[i].x - triangle_points[0].x) > eps)
                    && (abs(resized_loc_points[i].x - triangle_points[1].x) > eps)
                    && (abs(resized_loc_points[i].x - triangle_points[2].x) > eps))
            {
                return false;
            }
        }
    }
    return true;
}

bool QRDetectMulti::compareSquare::operator()(const Vec3i& a, const Vec3i& b) const
{
    Point2f a0 = points[a[0]];
    Point2f a1 = points[a[1]];
    Point2f a2 = points[a[2]];
    Point2f b0 = points[b[0]];
    Point2f b1 = points[b[1]];
    Point2f b2 = points[b[2]];
    return fabs((a1.x - a0.x) * (a2.y - a0.y) - (a2.x - a0.x) * (a1.y - a0.y)) <
           fabs((b1.x - b0.x) * (b2.y - b0.y) - (b2.x - b0.x) * (b1.y - b0.y));
}

int QRDetectMulti::findNumberLocalizationPoints(vector<Point2f>& tmp_localization_points)
{
    size_t number_possible_purpose = 1;
    if (purpose == SHRINKING)
        number_possible_purpose = 2;
    Mat tmp_shrinking = bin_barcode;
    int tmp_num_points = 0;
    int num_points = -1;
    for (eps_horizontal = 0.1; eps_horizontal < 0.4; eps_horizontal += 0.1)
    {
        tmp_num_points = 0;
        num_points = -1;
        if (purpose == SHRINKING)
            number_possible_purpose = 2;
        else
            number_possible_purpose = 1;
        for (size_t k = 0; k < number_possible_purpose; k++)
        {
            if (k == 1)
                bin_barcode = bin_barcode_fullsize;
            vector<Vec3d> list_lines_x = searchHorizontalLines();
            if (list_lines_x.empty())
            {
                if (k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty())
                        break;
                }
                else
                    break;
            }
            vector<Point2f> list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
            if (list_lines_y.size() < 3)
            {
                if (k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty())
                        break;
                    list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
                    if (list_lines_y.size() < 3)
                        break;
                }
                else
                    break;
            }
            vector<int> index_list_lines_y;
            for (size_t i = 0; i < list_lines_y.size(); i++)
                index_list_lines_y.push_back(-1);
            num_points = 0;
            for (size_t i = 0; i < list_lines_y.size() - 1; i++)
            {
                for (size_t j = i; j < list_lines_y.size(); j++ )
                {

                    double points_distance = norm(list_lines_y[i] - list_lines_y[j]);
                    if (points_distance <= 10)
                    {
                        if ((index_list_lines_y[i] == -1) && (index_list_lines_y[j] == -1))
                        {
                            index_list_lines_y[i] = num_points;
                            index_list_lines_y[j] = num_points;
                            num_points++;
                        }
                        else if (index_list_lines_y[i] != -1)
                            index_list_lines_y[j] = index_list_lines_y[i];
                        else if (index_list_lines_y[j] != -1)
                            index_list_lines_y[i] = index_list_lines_y[j];
                    }
                }
            }
            for (size_t i = 0; i < index_list_lines_y.size(); i++)
            {
                if (index_list_lines_y[i] == -1)
                {
                    index_list_lines_y[i] = num_points;
                    num_points++;
                }
            }
            if ((tmp_num_points < num_points) && (k == 1))
            {
                purpose = UNCHANGED;
                tmp_num_points = num_points;
                bin_barcode = bin_barcode_fullsize;
                coeff_expansion = 1.0;
            }
            if ((tmp_num_points < num_points) && (k == 0))
            {
                tmp_num_points = num_points;
            }
        }

        if ((tmp_num_points < 3) && (tmp_num_points >= 1))
        {
            const double min_side = std::min(bin_barcode_fullsize.size().width, bin_barcode_fullsize.size().height);
            if (min_side > 512)
            {
                bin_barcode = tmp_shrinking;
                purpose = SHRINKING;
                coeff_expansion = min_side / 512.0;
            }
            if (min_side < 512)
            {
                bin_barcode = tmp_shrinking;
                purpose = ZOOMING;
                coeff_expansion = 512 / min_side;
            }
        }
        else
            break;
    }
    if (purpose == SHRINKING)
        bin_barcode = tmp_shrinking;
    num_points = tmp_num_points;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if (list_lines_x.empty())
        return num_points;
    vector<Point2f> list_lines_y = extractVerticalLines(list_lines_x, eps_horizontal);
    if (list_lines_y.size() < 3)
        return num_points;
    if (num_points < 3)
        return num_points;

    Mat labels;
    kmeans(list_lines_y, num_points, labels,
            TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
            num_points, KMEANS_PP_CENTERS, tmp_localization_points);
    bin_barcode_temp = bin_barcode.clone();
    if (purpose == SHRINKING)
    {
        const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
        const int height = cvRound(bin_barcode.size().height * coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
    }
    else if (purpose == ZOOMING)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
    }
    else
    {
        bin_barcode = bin_barcode_fullsize.clone();
    }
    return num_points;
}

void QRDetectMulti::findQRCodeContours(vector<Point2f>& tmp_localization_points,
                                      vector< vector< Point2f > >& true_points_group, const int& num_qrcodes)
{
    Mat gray, blur_image, threshold_output;
    Mat bar = barcode;
    const int width  = cvRound(bin_barcode.size().width);
    const int height = cvRound(bin_barcode.size().height);
    Size new_size(width, height);
    resize(bar, bar, new_size, 0, 0, INTER_LINEAR);
    blur(bar, blur_image, Size(3, 3));
    threshold(blur_image, threshold_output, 50, 255, THRESH_BINARY);

    vector< vector< Point > > contours;
    vector<Vec4i> hierarchy;
    findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<Point2f> all_contours_points;
    for (size_t i = 0; i < contours.size(); i++)
    {
        for (size_t j = 0; j < contours[i].size(); j++)
        {
            all_contours_points.push_back(contours[i][j]);
        }
    }
    Mat qrcode_labels;
    vector<Point2f> clustered_localization_points;
    int count_contours = num_qrcodes;
    if (all_contours_points.size() < size_t(num_qrcodes))
        count_contours = (int)all_contours_points.size();
    kmeans(all_contours_points, count_contours, qrcode_labels,
          TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
          count_contours, KMEANS_PP_CENTERS, clustered_localization_points);

    vector< vector< Point2f > > qrcode_clusters(count_contours);
    for (int i = 0; i < count_contours; i++)
        for (int j = 0; j < int(all_contours_points.size()); j++)
        {
            if (qrcode_labels.at<int>(j, 0) == i)
            {
                qrcode_clusters[i].push_back(all_contours_points[j]);
            }
        }
    vector< vector< Point2f > > hull(count_contours);
    for (size_t i = 0; i < qrcode_clusters.size(); i++)
        convexHull(Mat(qrcode_clusters[i]), hull[i]);
    not_resized_loc_points = tmp_localization_points;
    resized_loc_points = tmp_localization_points;
    if (purpose == SHRINKING)
    {
        for (size_t j = 0; j < not_resized_loc_points.size(); j++)
        {
            not_resized_loc_points[j] *= coeff_expansion;
        }
    }
    else if (purpose == ZOOMING)
    {
        for (size_t j = 0; j < not_resized_loc_points.size(); j++)
        {
            not_resized_loc_points[j] /= coeff_expansion;
        }
    }

    true_points_group.resize(hull.size());

    for (size_t j = 0; j < hull.size(); j++)
    {
        for (size_t i = 0; i < not_resized_loc_points.size(); i++)
        {
            if (pointPolygonTest(hull[j], not_resized_loc_points[i], true) > 0)
            {
                true_points_group[j].push_back(tmp_localization_points[i]);
                tmp_localization_points[i].x = -1;
            }

        }
    }
    vector<Point2f> copy;
    for (size_t j = 0; j < tmp_localization_points.size(); j++)
    {
       if (tmp_localization_points[j].x != -1)
            copy.push_back(tmp_localization_points[j]);
    }
    tmp_localization_points = copy;
}

bool QRDetectMulti::checkSets(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& true_points_group_copy,
                              vector<Point2f>& tmp_localization_points)
{
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        if (true_points_group[i].size() < 3)
        {
            for (size_t j = 0; j < true_points_group[i].size(); j++)
                tmp_localization_points.push_back(true_points_group[i][j]);
            true_points_group[i].clear();
        }
    }
    vector< vector< Point2f > > temp_for_copy;
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        if (true_points_group[i].size() != 0)
            temp_for_copy.push_back(true_points_group[i]);
    }
    true_points_group = temp_for_copy;
    if (true_points_group.size() == 0)
    {
        true_points_group.push_back(tmp_localization_points);
        tmp_localization_points.clear();
    }
    if (true_points_group.size() == 0)
        return false;
    if (true_points_group[0].size() < 3)
        return false;


    vector<int> set_size(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        set_size[i] = int( (true_points_group[i].size() - 2 ) * (true_points_group[i].size() - 1) * true_points_group[i].size()) / 6;
    }

    vector< vector< Vec3i > > all_points(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
        all_points[i].resize(set_size[i]);
    int cur_cluster = 0;
    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        cur_cluster = 0;
        for (size_t l = 0; l < true_points_group[i].size() - 2; l++)
            for (size_t j = l + 1; j < true_points_group[i].size() - 1; j++)
                for (size_t k = j + 1; k < true_points_group[i].size(); k++)
                {
                    all_points[i][cur_cluster][0] = int(l);
                    all_points[i][cur_cluster][1] = int(j);
                    all_points[i][cur_cluster][2] = int(k);
                    cur_cluster++;
                }
    }

    for (size_t i = 0; i < true_points_group.size(); i++)
    {
        std::sort(all_points[i].begin(), all_points[i].end(), compareSquare(true_points_group[i]));
    }
    if (true_points_group.size() == 1)
    {
        int check_number = 35;
        if (set_size[0] > check_number)
            set_size[0] = check_number;
        all_points[0].resize(set_size[0]);
    }
    int iter = (int)localization_points.size();
    localization_points.resize(iter + true_points_group.size());
    transformation_points.resize(iter + true_points_group.size());

    true_points_group_copy = true_points_group;
    vector<int> end(true_points_group.size());
    for (size_t i = 0; i < true_points_group.size(); i++)
        end[i] = iter + set_size[i];
    ParallelSearch parallelSearch(true_points_group,
            true_points_group_copy, iter, end, all_points, *this);
    parallel_for_(Range(0, (int)true_points_group.size()), parallelSearch);

    return true;
}

void QRDetectMulti::deleteUsedPoints(vector<vector<Point2f> >& true_points_group, vector<vector<Point2f> >& loc,
                                     vector<Point2f>& tmp_localization_points)
{
    size_t iter = localization_points.size() - true_points_group.size() ;
    for (size_t s = 0; s < true_points_group.size(); s++)
    {
        if (localization_points[iter + s].empty())
            loc[s][0].x = -2;

        if (loc[s].size() == 3)
        {

            if ((true_points_group.size() > 1) || ((true_points_group.size() == 1) && (tmp_localization_points.size() != 0)) )
            {
                for (size_t j = 0; j < true_points_group[s].size(); j++)
                {
                    if (loc[s][j].x != -1)
                    {
                        loc[s][j].x = -1;
                        tmp_localization_points.push_back(true_points_group[s][j]);
                    }
                }
            }
        }
        vector<Point2f> for_copy;
        for (size_t j = 0; j < loc[s].size(); j++)
        {
            if ((loc[s][j].x != -1) && (loc[s][j].x != -2) )
            {
                for_copy.push_back(true_points_group[s][j]);
            }
            if ((loc[s][j].x == -2) && (true_points_group.size() > 1))
            {
                tmp_localization_points.push_back(true_points_group[s][j]);
            }
        }
        true_points_group[s] = for_copy;
    }

    vector< vector< Point2f > > for_copy_loc;
    vector< vector< Point2f > > for_copy_trans;


    for (size_t i = 0; i < localization_points.size(); i++)
    {
        if ((localization_points[i].size() == 3) && (transformation_points[i].size() == 4))
        {
            for_copy_loc.push_back(localization_points[i]);
            for_copy_trans.push_back(transformation_points[i]);
        }
    }
    localization_points = for_copy_loc;
    transformation_points = for_copy_trans;
}

bool QRDetectMulti::localization()
{
    CV_TRACE_FUNCTION();
    vector<Point2f> tmp_localization_points;
    int num_points = findNumberLocalizationPoints(tmp_localization_points);
    if (num_points < 3)
        return false;
    int num_qrcodes = divUp(num_points, 3);
    vector<vector<Point2f> > true_points_group;
    findQRCodeContours(tmp_localization_points, true_points_group, num_qrcodes);
    for (int q = 0; q < num_qrcodes; q++)
    {
       vector<vector<Point2f> > loc;
       size_t iter = localization_points.size();

       if (!checkSets(true_points_group, loc, tmp_localization_points))
            break;
       deleteUsedPoints(true_points_group, loc, tmp_localization_points);
       if ((localization_points.size() - iter) == 1)
           q--;
       if (((localization_points.size() - iter) == 0) && (tmp_localization_points.size() == 0) && (true_points_group.size() == 1) )
            break;
    }
    if ((transformation_points.size() == 0) || (localization_points.size() == 0))
       return false;
    return true;
}

bool QRDetectMulti::computeTransformationPoints(const size_t cur_ind)
{
    CV_TRACE_FUNCTION();

    if (localization_points[cur_ind].size() != 3)
    {
        return false;
    }

    vector<Point> locations, non_zero_elem[3], newHull;
    vector<Point2f> new_non_zero_elem[3];
    for (size_t i = 0; i < 3 ; i++)
    {
        Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
        uint8_t next_pixel, future_pixel = 255;
        int localization_point_x = cvRound(localization_points[cur_ind][i].x);
        int localization_point_y = cvRound(localization_points[cur_ind][i].y);
        int count_test_lines = 0, index = localization_point_x;
        for (; index < bin_barcode.cols - 1; index++)
        {
            next_pixel = bin_barcode.at<uint8_t>(localization_point_y, index + 1);
            if (next_pixel == future_pixel)
            {
                future_pixel = static_cast<uint8_t>(~future_pixel);
                count_test_lines++;

                if (count_test_lines == 2)
                {
                    // TODO avoid drawing functions
                    floodFill(bin_barcode, mask,
                            Point(index + 1, localization_point_y), 255,
                            0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                    break;
                }
            }

        }
        Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
        findNonZero(mask_roi, non_zero_elem[i]);
        newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
    }
    convexHull(newHull, locations);
    for (size_t i = 0; i < locations.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < non_zero_elem[j].size(); k++)
            {
                if (locations[i] == non_zero_elem[j][k])
                {
                    new_non_zero_elem[j].push_back(locations[i]);
                }
            }
        }
    }

    if (new_non_zero_elem[0].size() == 0)
        return false;

    double pentagon_diag_norm = -1;
    Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
        {
            double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
            if (temp_norm > pentagon_diag_norm)
            {
                down_left_edge_point = new_non_zero_elem[1][i];
                up_right_edge_point  = new_non_zero_elem[2][j];
                pentagon_diag_norm = temp_norm;
            }
        }
    }

    if (down_left_edge_point == Point2f(0, 0) ||
        up_right_edge_point  == Point2f(0, 0))
    {
        return false;
    }

    double max_area = -1;
    up_left_edge_point = new_non_zero_elem[0][0];

    for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
    {
        vector<Point2f> list_edge_points;
        list_edge_points.push_back(new_non_zero_elem[0][i]);
        list_edge_points.push_back(down_left_edge_point);
        list_edge_points.push_back(up_right_edge_point);

        double temp_area = fabs(contourArea(list_edge_points));
        if (max_area < temp_area)
        {
            up_left_edge_point = new_non_zero_elem[0][i];
            max_area = temp_area;
        }
    }

    Point2f down_max_delta_point, up_max_delta_point;
    double norm_down_max_delta = -1, norm_up_max_delta = -1;
    for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i]) + norm(down_left_edge_point - new_non_zero_elem[1][i]);
        if (norm_down_max_delta < temp_norm_delta)
        {
            down_max_delta_point = new_non_zero_elem[1][i];
            norm_down_max_delta = temp_norm_delta;
        }
    }


    for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
    {
        double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i]) + norm(up_right_edge_point - new_non_zero_elem[2][i]);
        if (norm_up_max_delta < temp_norm_delta)
        {
            up_max_delta_point = new_non_zero_elem[2][i];
            norm_up_max_delta = temp_norm_delta;
        }
    }
    vector<Point2f> tmp_transformation_points;
    tmp_transformation_points.push_back(down_left_edge_point);
    tmp_transformation_points.push_back(up_left_edge_point);
    tmp_transformation_points.push_back(up_right_edge_point);
    tmp_transformation_points.push_back(intersectionLines(
                    down_left_edge_point, down_max_delta_point,
                    up_right_edge_point, up_max_delta_point));
    transformation_points[cur_ind] = tmp_transformation_points;

    vector<Point2f> quadrilateral = getQuadrilateral(transformation_points[cur_ind]);
    transformation_points[cur_ind] = quadrilateral;

    return true;
}

bool QRCodeDetector::detectMulti(InputArray in, OutputArray points) const
{
    Mat inarr;
    if (!checkQRInputImage(in, inarr))
    {
        points.release();
        return false;
    }

    QRDetectMulti qrdet;
    qrdet.init(inarr, p->epsX, p->epsY);
    if (!qrdet.localization())
    {
        points.release();
        return false;
    }
    vector< vector< Point2f > > pnts2f = qrdet.getTransformationPoints();
    vector<Point2f> trans_points;
    for(size_t i = 0; i < pnts2f.size(); i++)
        for(size_t j = 0; j < pnts2f[i].size(); j++)
            trans_points.push_back(pnts2f[i][j]);

    updatePointsResult(points, trans_points);

    return true;
}

class ParallelDecodeProcess : public ParallelLoopBody
{
public:
    ParallelDecodeProcess(Mat& inarr_, vector<QRDecode>& qrdec_, vector<std::string>& decoded_info_,
            vector<Mat>& straight_barcode_, vector< vector< Point2f > >& src_points_)
        : inarr(inarr_), qrdec(qrdec_), decoded_info(decoded_info_)
        , straight_barcode(straight_barcode_), src_points(src_points_)
    {
        // nothing
    }
    void operator()(const Range& range) const CV_OVERRIDE
    {
        for (int i = range.start; i < range.end; i++)
        {
            qrdec[i].init(inarr, src_points[i]);
            bool ok = qrdec[i].fullDecodingProcess();
            if (ok)
            {
                decoded_info[i] = qrdec[i].getDecodeInformation();
                straight_barcode[i] = qrdec[i].getStraightBarcode();
            }
            else if (std::min(inarr.size().width, inarr.size().height) > 512)
            {
                const int min_side = std::min(inarr.size().width, inarr.size().height);
                double coeff_expansion = min_side / 512;
                const int width  = cvRound(inarr.size().width  / coeff_expansion);
                const int height = cvRound(inarr.size().height / coeff_expansion);
                Size new_size(width, height);
                Mat inarr2;
                resize(inarr, inarr2, new_size, 0, 0, INTER_AREA);
                for (size_t j = 0; j < 4; j++)
                {
                    src_points[i][j] /= static_cast<float>(coeff_expansion);
                }
                qrdec[i].init(inarr2, src_points[i]);
                ok = qrdec[i].fullDecodingProcess();
                if (ok)
                {
                    decoded_info[i] = qrdec[i].getDecodeInformation();
                    straight_barcode[i] = qrdec[i].getStraightBarcode();
                }
            }
            if (decoded_info[i].empty())
                decoded_info[i] = "";
        }
    }

private:
    Mat& inarr;
    vector<QRDecode>& qrdec;
    vector<std::string>& decoded_info;
    vector<Mat>& straight_barcode;
    vector< vector< Point2f > >& src_points;

};

bool QRCodeDetector::decodeMulti(
        InputArray img,
        InputArray points,
        CV_OUT std::vector<cv::String>& decoded_info,
        OutputArrayOfArrays straight_qrcode
    ) const
{
    Mat inarr;
    if (!checkQRInputImage(img, inarr))
        return false;
    CV_Assert(points.size().width > 0);
    CV_Assert((points.size().width % 4) == 0);
    vector< vector< Point2f > > src_points ;
    Mat qr_points = points.getMat();
    qr_points = qr_points.reshape(2, 1);
    for (int i = 0; i < qr_points.size().width ; i += 4)
    {
        vector<Point2f> tempMat = qr_points.colRange(i, i + 4);
        if (contourArea(tempMat) > 0.0)
        {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(src_points.size() > 0);
    vector<QRDecode> qrdec(src_points.size());
    vector<Mat> straight_barcode(src_points.size());
    vector<std::string> info(src_points.size());
    ParallelDecodeProcess parallelDecodeProcess(inarr, qrdec, info, straight_barcode, src_points);
    parallel_for_(Range(0, int(src_points.size())), parallelDecodeProcess);
    vector<Mat> for_copy;
    for (size_t i = 0; i < straight_barcode.size(); i++)
    {
        if (!(straight_barcode[i].empty()))
            for_copy.push_back(straight_barcode[i]);
    }
    straight_barcode = for_copy;
    vector<Mat> tmp_straight_qrcodes;
    if (straight_qrcode.needed())
    {
        for (size_t i = 0; i < straight_barcode.size(); i++)
        {
            Mat tmp_straight_qrcode;
            tmp_straight_qrcodes.push_back(tmp_straight_qrcode);
            straight_barcode[i].convertTo(((OutputArray)tmp_straight_qrcodes[i]),
                                             ((OutputArray)tmp_straight_qrcodes[i]).fixedType() ?
                                             ((OutputArray)tmp_straight_qrcodes[i]).type() : CV_32FC2);
        }
        straight_qrcode.createSameSize(tmp_straight_qrcodes, CV_32FC2);
        straight_qrcode.assign(tmp_straight_qrcodes);
    }
    decoded_info.clear();
    for (size_t i = 0; i < info.size(); i++)
    {
       decoded_info.push_back(info[i]);
    }
    if (!decoded_info.empty())
        return true;
    else
        return false;
}

bool QRCodeDetector::detectAndDecodeMulti(
        InputArray img,
        CV_OUT std::vector<cv::String>& decoded_info,
        OutputArray points_,
        OutputArrayOfArrays straight_qrcode
    ) const
{
    Mat inarr;
    if (!checkQRInputImage(img, inarr))
    {
        points_.release();
        return false;
    }

    vector<Point2f> points;
    bool ok = detectMulti(inarr, points);
    if (!ok)
    {
        points_.release();
        return false;
    }
    updatePointsResult(points_, points);
    decoded_info.clear();
    ok = decodeMulti(inarr, points, decoded_info, straight_qrcode);
    return ok;
}

}  // namespace

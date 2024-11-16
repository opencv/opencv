// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Copyright 2008 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef QBAR_AI_QBAR_QBARSTRUCT_H_
#define QBAR_AI_QBAR_QBARSTRUCT_H_

#include <stdint.h>
#include <string>
#include <vector>
#include <memory>
#define QBAR_VERSION "3.2.20190712"

namespace cv {
/////////////////////////////////////////// config struct
// system config
enum QBAR_SEARCH_MODE
{
    SEARCH_ONE = 0,
    SEARCH_MULTI = 1,
};

enum QBAR_SCAN_MODE{
    SCAN_VIDEO = 0,
    SCAN_FILE = 1,
};

enum PixelFormat{
    PIX_FMT_BGRA8888,
    PIX_FMT_BGR888,
    PIX_FMT_RGBA8888,
    PIX_FMT_RGB888,
    PIX_FMT_GRAY,
};

struct QBAR_ML_MODE
{
    std::string detection_model_path_;
    std::string detection_param_path_;
    std::string super_resolution_model_path_;
    std::string super_resolution_param_path_;
    std::string qbar_segmentation_model_path_;
    std::string qbar_segmentation_param_path_;

    enum CLASSIFY_ID
    {
        CLASS_QRCODE = 1,
        CLASS_PDF417 = 2,
        CLASS_ONED   = 3,
        CLASS_DATAMATRIX = 4,
        CLASS_QRCODE_OR_DM = 5,
    };
};

struct QBAR_OPT{
    bool OPT_SR = false;
    bool OPT_DET = false;
    bool OPT_LIBDMTX = false;
    bool OPT_FORCEDM = true;
};

struct QBAR_MODE
{	
    QBAR_SEARCH_MODE searchMode;
    QBAR_SCAN_MODE scanMode;
    std::string inputCharset;
    std::string outputCharset;

    bool enable_time_consuming_log;  // option: just for log or test
    bool enable_time_consuming;
    bool useAI;
    bool scan_multi_online;
    bool enable_barcode_seg;

    QBAR_OPT opt;

    // for ai
    QBAR_ML_MODE qbar_ml_mode;

    QBAR_MODE(QBAR_SEARCH_MODE search=SEARCH_ONE, QBAR_SCAN_MODE scan = SCAN_VIDEO):
    searchMode(search), scanMode(scan), enable_time_consuming_log(false), enable_time_consuming(true), useAI(true), scan_multi_online(false), enable_barcode_seg(false), opt(QBAR_OPT())
    {};
};

// reader config, if not set, try all reader
enum QBAR_READER{
    ONED_BARCODE = 1, // barcode, which includes UPC_A, UPC_E, EAN_8, EAN_13, CODE_39, CODE_93, CODE_128, ITF, CODABAR
    QRCODE = 2, // QRCODE
    PDF417 = 3, // PDF417
    DATAMATRIX = 4, // DATAMATRIX
};

///////////////////////////// result struct
struct QBAR_POINT 
{
    float x, y;
    QBAR_POINT() {
        x = 0;
        y = 0;
    }
    QBAR_POINT(float x_, float y_): x(x_), y(y_) {}
};
struct QBAR_AREA
{
    int x, y, width, height;
    QBAR_AREA() {
        x = 0;
        y = 0;
        width = 0;
        height = 0;
    }
};

struct QBAR_REPORT_MSG
{
    int qrcodeVersion;  // version of qrcode(0-40), -1 for barcode and pdf417
    int pyramidLv;  // pyramid level
    std::string binaryMethod;  // the binarization method used
    std::string ecLevel;
    std::string charsetMode;
    std::string scale_list_;
    float decode_scale_;
    uint32_t detect_time_;
    uint32_t sr_time_;
    bool has_sr;
    uint32_t decode_time_;
    bool in_white_list_;
    bool in_black_list_;
    
    uint32_t pre_detect_time_;
    uint32_t detect_infer_pre_time_;
    uint32_t detect_infer_time_;
    uint32_t detect_infer_after_time_;
    uint32_t after_detect_time_;
    uint32_t seg_time_;
    bool has_seg;
    uint32_t after_seg_time_;
    uint32_t decode_all_time_;
    bool has_decode;
    
    QBAR_REPORT_MSG() : qrcodeVersion(-1), pyramidLv(-1), binaryMethod(""), ecLevel("0"),
    scale_list_(""), decode_scale_(0.0), detect_time_(0), sr_time_(0), decode_time_(0.0),
    in_white_list_(false), in_black_list_(false),
    pre_detect_time_(0.0), detect_infer_pre_time_(0.0), detect_infer_time_(0.0), detect_infer_after_time_(0.0),
    after_detect_time_(0.0), seg_time_(0.0), after_seg_time_(0.0), decode_all_time_(0.0)
    {}
};

struct QBAR_RESULT
{
    int typeID = 0;
    std::string typeName;
    std::string data;
    std::string charset;
    std::vector<QBAR_POINT> points;
    QBAR_AREA area = QBAR_AREA();
    QBAR_REPORT_MSG reportMsg;
    int priorityLevel;  // 0-other, 1-white list, 2-black list

    static QBAR_RESULT MakeInvalid() {
        return QBAR_RESULT();
    }
};

struct QBAR_CODE_DETECT_INFO
{
    QBAR_READER readerId;
    std::vector<QBAR_POINT> points;
    float prob;
    // int clsId;
};

struct QBAR_ZOOM_INFO
{
    bool isZoom;
    float zoomFactor;
};

struct QBAR_INFO {
    int track_id;
    QBAR_CODE_DETECT_INFO detect_info;  //detect result
    QBAR_RESULT result_info;  // decode result
    float result_confidence;  // decode result confidence
    int info_frame_delay;
};

//////////////////////////////////////////////////// encode image

class QBAR_IMAGE
{
public:
    int width;  // dot matrix width
    int height;  // dot matrix height
    
    std::vector<uint8_t> data;
    
    uint8_t get(int x, int y)
    {
        return data.at(y*width + x);
    };
    
    void set(int x, int y, uint8_t value)
    {
        data.at(y*width + x) = value;
    }
};

enum QBAR_QRCODE_ERROR_LEVEL{
    L = 0,
    M = 1,
    Q = 2,
    H = 3,
};
enum QBAR_CODE_FORMAT{
    FMT_AZTEC = 1,
    FMT_CODABAR = 2,
    FMT_CODE39 = 3,
    FMT_CODE93 = 4,
    FMT_CODE128 = 5,
    FMT_DATAMATRIX = 6,
    FMT_EAN8 = 7,
    FMT_EAN13 = 8,
    FMT_ITF = 9,
    FMT_MAXICODE = 10,
    FMT_PDF417 = 11,
    FMT_QRCODE = 12,
    FMT_RSS14 = 13,
    FMT_RSSEXPANDED = 14,
    FMT_UPCA = 15,
    FMT_UPCE = 16,
    FMT_UPCEAN_EXTENSION = 17,
    FMT_CODE25 = 18,
};

/////////////////////////////////// debug
// enum QBAR_BINARIZER
// {
//     Hybrid = 0,
//     FastWindow = 1,
//     SimpleAdaptive = 2,
//     GlobalHistogram=3,
//     OTSU=4,
//     Niblack=5,
//     Adaptive=6,
//     HistogramBackground=7
// };


// struct QBAR_DEBUG
// {
//     QBAR_BINARIZER binarizer;
// };

enum QBAR_CONFIG_TYPE{
    CONFIG_RESERVED0 = 0,
    CONFIG_RESERVED1 = 1,
    CONFIG_RESERVED2 = 2,
    CONFIG_MAX_QBAR_OUTPUT_NUM = 3,
};


//======= For Encode =========//

enum QRCODE_EYE_SHAPE {
    EYE_SQUARE = 0,
    EYE_CIRCLE = 1,
    EYE_ROUND = 2,
};

enum QRCODE_MODULE_SHAPE {
    MODULE_SQUARE = 0,
    MODULE_CIRCLE = 1,
    MODULE_ROUND = 2,
};

enum QRCODE_IMAGE_MODE {
    IMAGE_GRAY = 0,
    IMAGE_COLOR = 1,
    IMAGE_ALPHA = 2,
};

struct QBAR_COLOR {
    int R, G, B;
    int count;
    
    QBAR_COLOR() {
        R = 0;
        G = 0;
        B = 0;
        count = 0;
    }
    QBAR_COLOR(int r, int g, int b) {
        R = r;
        G = g;
        B = b;
        
        count = 0;
    }
};

enum PersonalMode
{
    MODE_PERSONAL_COLOR = 0,
    MODE_PERSONAL_WHITE = 1,
    MODE_PERSONAL_BLACK = 2,
};

struct QBarImageInfo
{
    uint8_t* image_data_;
    int width_;
    int height_;
    PixelFormat image_format_;
};

struct PersonalParam
{
#ifdef USE_IMREAD
    std::string eye_path_;
    std::string black_eye_path_;
    std::string module_path_;
    std::string black_module_path_;
    std::string logo_path_;
    std::string black_logo_path_;
#endif
    
    QBarImageInfo eye_image_;
    QBarImageInfo black_eye_image_;
    QBarImageInfo module_image_;
    QBarImageInfo black_module_image_;
    QBarImageInfo logo_image_;
    QBarImageInfo black_logo_image_;
    
    PersonalMode mode_;
    
    QBarImageInfo head_image_;
    bool is_group_qrcode_;
    
    PersonalParam () {
        mode_ = MODE_PERSONAL_BLACK;
        is_group_qrcode_ = false;
    }
};

enum QBAR_SOURCE {
    SOURCE_PERSONALQR = 0,
    SOURCE_NORMAL = 1,
};

struct QBarDrawParam {
    QRCODE_EYE_SHAPE eye_shape_;
    QRCODE_MODULE_SHAPE module_shape_;
    QBAR_SOURCE source_;
    QRCODE_IMAGE_MODE image_mode_;
    
    QBarImageInfo logo_info_;
    QBAR_COLOR fg_color_;
    QBAR_COLOR bg_color_;
    
    int module_size_;
    
    QBarDrawParam() {
        eye_shape_ = EYE_SQUARE;
        module_shape_ = MODULE_SQUARE;
        image_mode_ = IMAGE_COLOR;
        fg_color_ = QBAR_COLOR(0, 0, 0);
        bg_color_ = QBAR_COLOR(255, 255, 255);
        module_size_ = 20;
    }
};
}  // namespace cv
#endif  // QBAR_AI_QBAR_QBARSTRUCT_H_
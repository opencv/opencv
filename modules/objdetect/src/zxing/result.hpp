#ifndef __RESULT_H__
#define __RESULT_H__

/*
 *  Result.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
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

#include <string>
#include "common/array.hpp"
#include "common/counted.hpp"
#include "common/str.hpp"
#include "result_point.hpp"
#include "barcode_format.hpp"
#include <stdint.h>

namespace zxing {

class Result : public Counted {
private:
    Ref<String> text_;
    ArrayRef<char> rawBytes_;
    ArrayRef< Ref<ResultPoint> > resultPoints_;
    BarcodeFormat format_;
    std::string charset_;
    
    // chango add for report 20160323
    int qrcodeVersion_;
    int pyramidLv_;
    int binaryMethod_;
    std::string ecLevel_;
    std::string charsetMode_;
    std::string scale_list_;
    float decode_scale_;
    uint32_t detect_time_;
    uint32_t sr_time_;
    bool has_sr_;
    uint32_t decode_time_;
    
    uint32_t pre_detect_time_;
    uint32_t detect_infer_pre_time_;
    uint32_t detect_infer_time_;
    uint32_t detect_infer_after_time_;
    uint32_t after_detect_time_;
    uint32_t seg_time_;
    bool has_seg_ = false;
    uint32_t after_seg_time_;
    uint32_t decode_all_time_;
    bool has_decode_ = false;
    
#ifdef CALC_CODE_AREA_SCORE
    float codeAreaScore_;
    bool mixMode_;
#endif
    
public:
    Result(Ref<String> text,
           ArrayRef<char> rawBytes,
           ArrayRef< Ref<ResultPoint> > resultPoints,
           BarcodeFormat format);
    
    Result(Ref<String> text,
           ArrayRef<char> rawBytes,
           ArrayRef< Ref<ResultPoint> > resultPoints,
           BarcodeFormat format,
           std::string charset);
    
    Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints, BarcodeFormat format, std::string charset, int QRCodeVersion, std::string ecLevel, std::string charsetMode);
    
    ~Result();
    
    Ref<String> getText();
    ArrayRef<char> getRawBytes();
    ArrayRef< Ref<ResultPoint> > const& getResultPoints() const;
    ArrayRef< Ref<ResultPoint> >& getResultPoints();
    void setResultPoints(int idx, float x, float y);
    BarcodeFormat getBarcodeFormat() const;
    std::string getCharset() const;
    std::string getChartsetMode() const;
    void enlargeResultPoints(int scale);
    
    int getQRCodeVersion() const { return qrcodeVersion_; };
    void setQRCodeVersion(int QRCodeVersion) { qrcodeVersion_ = QRCodeVersion; };
    int getPyramidLv() const { return pyramidLv_; };
    void setPyramidLv(int pyramidLv) { pyramidLv_ = pyramidLv; };
    int getBinaryMethod() const { return binaryMethod_; };
    void setBinaryMethod(int binaryMethod) { binaryMethod_ = binaryMethod; };
    std::string getEcLevel() const { return ecLevel_; }
    void setEcLevel(char ecLevel) { ecLevel_ = ecLevel; }
    std::string getScaleList() {return scale_list_;};
    void setScaleList(const std::string & scale_list) {scale_list_ = scale_list;};
    float getDecodeScale()  {return decode_scale_;};
    void setDecodeScale(float decode_scale)   {decode_scale_ = decode_scale;};
    uint32_t getDetectInferPreTime()    {return detect_infer_pre_time_;};
    void setDetectInferPreTime(uint32_t detect_infer_pre_time)    {detect_infer_pre_time_ = detect_infer_pre_time;};
    uint32_t getDetectInferTime()    {return detect_infer_time_;};
    void setDetectInferTime(uint32_t detect_infer_time)    {detect_infer_time_ = detect_infer_time;};
    uint32_t getDetectInferAfterTime()    {return detect_infer_after_time_;};
    void setDetectInferAfterTime(uint32_t detect_infer_after_time)    {detect_infer_after_time_ = detect_infer_after_time;};
    uint32_t getDetectTime()    {return detect_time_;};
    void setDetectTime(uint32_t detect_time)    {detect_time_ = detect_time;};
    uint32_t getSrTime()  {return sr_time_;};
    void setSrTime(uint32_t sr_time)    {sr_time_ = sr_time;};
    void setHasSr(bool has_sr)    { has_sr_ = has_sr;};
    bool getHasSr() { return has_sr_; }
    uint32_t getDecodeTime() {return decode_time_;}
    void setDecodeTime(uint32_t decode_time) {decode_time_ = decode_time;}
    
    void setPreDetectTime(uint32_t pre_detect_time) {pre_detect_time_ = pre_detect_time;}
    uint32_t getPreDetectTime() {return pre_detect_time_;}
    void setAfterDetectTime(uint32_t after_detect_time) {after_detect_time_ = after_detect_time;}
    uint32_t getAfterDetectTime() {return after_detect_time_;}
    void setSegTime(uint32_t seg_time) {seg_time_ = seg_time;}
    void setHasSeg(bool has_seg) { has_seg_ = has_seg;}
    bool getHasSeg() { return has_seg_; }
    uint32_t getSegeTime() {return seg_time_;}
    void setAfterSegTime(uint32_t after_seg_time) {after_seg_time_ = after_seg_time;}
    uint32_t getAfterSegTime() {return after_seg_time_;}
    void setDecodeAllTime(uint32_t decode_all_time) {decode_all_time_ = decode_all_time;}
    uint32_t getDecodeAllTime() {return decode_all_time_;}
    void setHasDecode(bool has_decode) { has_decode_ = has_decode;}
    bool getHasDecode() { return has_decode_; }
    
#ifdef CALC_CODE_AREA_SCORE
    void setCodeAreaScore(float codeAreaScore) {codeAreaScore_ = codeAreaScore;}
    float getCodeAreaScore() const{ return codeAreaScore_;}
    void setMixMode(bool mixMode) {mixMode_ = mixMode; }
    bool getMixMode() { return mixMode_; }
    
#endif
    
    friend std::ostream& operator<<(std::ostream &out, Result& result);
};

}  // namespace zxing
#endif  // __RESULT_H__

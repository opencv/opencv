/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"

namespace cv
{

class AlignMTBImpl : public AlignMTB
{
public:
    AlignMTBImpl(int max_bits, int exclude_range, bool cut) :
        max_bits(max_bits),
        exclude_range(exclude_range),
        cut(cut),
        name("AlignMTB")
    {
    }
    
    void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                 const std::vector<float>& times, InputArray response)
    {
        process(src, dst);
    }

    void process(InputArrayOfArrays _src, std::vector<Mat>& dst)
    {
        std::vector<Mat> src;
        _src.getMatVector(src);
        
        checkImageDimensions(src);
        dst.resize(src.size());

        size_t pivot = src.size() / 2;
        dst[pivot] = src[pivot];
        Mat gray_base;
        cvtColor(src[pivot], gray_base, COLOR_RGB2GRAY);
        std::vector<Point> shifts;

        for(size_t i = 0; i < src.size(); i++) {
            if(i == pivot) {
                shifts.push_back(Point(0, 0));
                continue;
            }
            Mat gray;
            cvtColor(src[i], gray, COLOR_RGB2GRAY);
            Point shift;
            calculateShift(gray_base, gray, shift);
            shifts.push_back(shift);
            shiftMat(src[i], dst[i], shift);
        }
        if(cut) {
            Point max(0, 0), min(0, 0);
            for(size_t i = 0; i < shifts.size(); i++) {
                if(shifts[i].x > max.x) {
                    max.x = shifts[i].x;
                }
                if(shifts[i].y > max.y) {
                    max.y = shifts[i].y;
                }
                if(shifts[i].x < min.x) {
                    min.x = shifts[i].x;
                }
                if(shifts[i].y < min.y) {
                    min.y = shifts[i].y;
                }
            }
            Point size = dst[0].size();
            for(size_t i = 0; i < dst.size(); i++) {
                dst[i] = dst[i](Rect(max, min + size));
            }
        }
    }

    void calculateShift(InputArray _img0, InputArray _img1, Point& shift)
    {
        Mat img0 = _img0.getMat();
        Mat img1 = _img1.getMat();
        CV_Assert(img0.channels() == 1 && img0.type() == img1.type());
        CV_Assert(img0.size() == img0.size());

        int maxlevel = static_cast<int>(log((double)max(img0.rows, img0.cols)) / log(2.0)) - 1;
        maxlevel = min(maxlevel, max_bits - 1);

        std::vector<Mat> pyr0;
        std::vector<Mat> pyr1;
        buildPyr(img0, pyr0, maxlevel);
        buildPyr(img1, pyr1, maxlevel);    
    
        shift = Point(0, 0);
        for(int level = maxlevel; level >= 0; level--) {
        
            shift *= 2;
            Mat tb1, tb2, eb1, eb2;
            computeBitmaps(pyr0[level], tb1, eb1);
            computeBitmaps(pyr1[level], tb2, eb2);

            int min_err = pyr0[level].total();
            Point new_shift(shift);
            for(int i = -1; i <= 1; i++) {
                for(int j = -1; j <= 1; j++) {
                    Point test_shift = shift + Point(i, j);
                    Mat shifted_tb2, shifted_eb2, diff;
                    shiftMat(tb2, shifted_tb2, test_shift);
                    shiftMat(eb2, shifted_eb2, test_shift);
                    bitwise_xor(tb1, shifted_tb2, diff);
                    bitwise_and(diff, eb1, diff);
                    bitwise_and(diff, shifted_eb2, diff);
                    int err = countNonZero(diff);
                    if(err < min_err) {
                        new_shift = test_shift;
                        min_err = err;
                    }        
                }
            }
            shift = new_shift;
        }
    }

    void shiftMat(InputArray _src, OutputArray _dst, const Point shift) 
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        Mat res = Mat::zeros(src.size(), src.type());
        int width = src.cols - abs(shift.x);
        int height = src.rows - abs(shift.y);
        Rect dst_rect(max(shift.x, 0), max(shift.y, 0), width, height);
        Rect src_rect(max(-shift.x, 0), max(-shift.y, 0), width, height);
        src(src_rect).copyTo(res(dst_rect));
        res.copyTo(dst);
    }

    int getMaxBits() const { return max_bits; }
    void setMaxBits(int val) { max_bits = val; }

    int getExcludeRange() const { return exclude_range; }
    void setExcludeRange(int val) { exclude_range = val; }

    bool getCut() const { return cut; }
    void setCut(bool val) { cut = val; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "max_bits" << max_bits
           << "exclude_range" << exclude_range 
           << "cut" << static_cast<int>(cut);
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        max_bits = fn["max_bits"];
        exclude_range = fn["exclude_range"];
        int cut_val = fn["cut"];
        cut = static_cast<bool>(cut_val);
    }

    void computeBitmaps(Mat& img, Mat& tb, Mat& eb)
    {
        int median = getMedian(img);
        compare(img, median, tb, CMP_GT);
        compare(abs(img - median), exclude_range, eb, CMP_GT);
    }

protected:
    String name;
    int max_bits, exclude_range;
    bool cut;

    void downsample(Mat& src, Mat& dst)
    {
        dst = Mat(src.rows / 2, src.cols / 2, CV_8UC1);

        int offset = src.cols * 2;
        uchar *src_ptr = src.ptr();
        uchar *dst_ptr = dst.ptr();
        for(int y = 0; y < dst.rows; y ++) {
            uchar *ptr = src_ptr;
            for(int x = 0; x < dst.cols; x++) {
                dst_ptr[0] = ptr[0];
                dst_ptr++;
                ptr += 2;
            }
            src_ptr += offset;
        }
    }

    void buildPyr(Mat& img, std::vector<Mat>& pyr, int maxlevel) 
    {
        pyr.resize(maxlevel + 1);
        pyr[0] = img.clone();
        for(int level = 0; level < maxlevel; level++) {
            downsample(pyr[level], pyr[level + 1]);
        }
    }

    int getMedian(Mat& img)
    {
        int channels = 0;
        Mat hist; 
        int hist_size = 256;
        float range[] = {0, 256} ;
        const float* ranges[] = {range};
        calcHist(&img, 1, &channels, Mat(), hist, 1, &hist_size, ranges);
        float *ptr = hist.ptr<float>();
        int median = 0, sum = 0;
        int thresh = img.total() / 2;
        while(sum < thresh && median < 256) {
            sum += static_cast<int>(ptr[median]);
            median++;
        }
        return median;
    }
};

Ptr<AlignMTB> createAlignMTB(int max_bits, int exclude_range, bool cut)
{
    return new AlignMTBImpl(max_bits, exclude_range, cut);
}

class floatIndexCmp {
public:
    floatIndexCmp(std::vector<float> data) :
      data(data)
    {
    }

    bool operator() (int i,int j) 
    { 
        return data[i] < data[j];
    }
protected:
    std::vector<float> data;
};

class GhostbusterOrderImpl : public GhostbusterOrder
{
public:
    GhostbusterOrderImpl(int underexp, int overexp) :
      underexp(underexp),
      overexp(overexp),
      name("GhostbusterOrder")
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times, Mat response) 
    {
        process(src, dst);
    }

    void process(InputArrayOfArrays src, OutputArray dst)
    {
        std::vector<Mat> unsorted_images;
        src.getMatVector(unsorted_images);
        checkImageDimensions(unsorted_images);

        std::vector<Mat> images;
        sortImages(unsorted_images, images);

        int channels = images[0].channels();
        dst.create(images[0].size(), CV_8U);

        Mat res = Mat::zeros(images[0].size(), CV_8U);

        std::vector<Mat> splitted(channels);
        split(images[0], splitted);
        for(int i = 0; i < images.size() - 1; i++) {
            
            std::vector<Mat> next_splitted(channels);
            split(images[i + 1], next_splitted);
            
            for(int c = 0; c < channels; c++) {
                Mat exposed = (splitted[c] >= underexp) & (splitted[c] <= overexp);
                exposed &= (next_splitted[c] >= underexp) & (next_splitted[c] <= overexp);
                Mat ghost = (splitted[c] > next_splitted[c]) & exposed;
                res |= ghost;
            }
            splitted = next_splitted;
        }
        res.copyTo(dst.getMat());
    }

    int getUnderexp() {return underexp;}
    void setUnderexp(int value) {underexp = value;}

    int getOverexp() {return overexp;}
    void setOverexp(int value) {overexp = value;}

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "overexp" << overexp
           << "underexp" << underexp;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        overexp = fn["overexp"];
        underexp = fn["underexp"];
    }

protected:
    int overexp, underexp;
    String name;

    void sortImages(std::vector<Mat>& images, std::vector<Mat>& sorted)
    {
        std::vector<int>indices(images.size());
        std::vector<float>means(images.size());
        for(size_t i = 0; i < images.size(); i++) {
            indices[i] = i;
            means[i] = mean(mean(images[i]))[0];
        }
        sort(indices.begin(), indices.end(), floatIndexCmp(means));
        sorted.resize(images.size());
        for(size_t i = 0; i < images.size(); i++) {
            sorted[i] = images[indices[i]];
        }
    }
};

Ptr<GhostbusterOrder> createGhostbusterOrder(int underexp, int overexp)
{
    return new GhostbusterOrderImpl(underexp, overexp);
}

class GhostbusterPredictImpl : public GhostbusterPredict
{
public:
    GhostbusterPredictImpl(int thresh, int underexp, int overexp) :
      thresh(thresh),
      underexp(underexp),
      overexp(overexp),
      name("GhostbusterPredict")
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times, Mat response)
    {
        std::vector<Mat> images;
        src.getMatVector(images);
        checkImageDimensions(images);

        int channels = images[0].channels();
        dst.create(images[0].size(), CV_8U);

        Mat res = Mat::zeros(images[0].size(), CV_8U);

        Mat radiance;
        LUT(images[0], response, radiance);
        std::vector<Mat> splitted(channels);
        split(radiance, splitted);
        std::vector<Mat> resp_split(channels);
        split(response, resp_split);
        for(int i = 0; i < images.size() - 1; i++) {
            
            std::vector<Mat> next_splitted(channels);
            LUT(images[i + 1], response, radiance);
            split(radiance, next_splitted);
            
            for(int c = 0; c < channels; c++) {

                Mat predicted = splitted[c] / times[i] * times[i + 1];

                Mat low = max(thresh, next_splitted[c]) - thresh;
                Mat high = min(255 - thresh, next_splitted[c]) + thresh;
                low.convertTo(low, CV_8U);
                high.convertTo(high, CV_8U);
                LUT(low, resp_split[c], low);
                LUT(high, resp_split[c], high);

                Mat exposed = (splitted[c] >= underexp) & (splitted[c] <= overexp);
                exposed &= (next_splitted[c] >= underexp) & (next_splitted[c] <= overexp);

                Mat ghost = (low < predicted) & (predicted < high);
                ghost &= exposed;
                res |= ghost;
            }
            splitted = next_splitted;
        }
        res.copyTo(dst.getMat());
    }

    virtual void process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times)
    {
        process(src, dst, times, linearResponse(3));
    }

    CV_WRAP virtual int getThreshold() {return thresh;}
    CV_WRAP virtual void setThreshold(int value) {thresh = value;}

    int getUnderexp() {return underexp;}
    void setUnderexp(int value) {underexp = value;}

    int getOverexp() {return overexp;}
    void setOverexp(int value) {overexp = value;}

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "overexp" << overexp
           << "underexp" << underexp
           << "thresh" << thresh;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        overexp = fn["overexp"];
        underexp = fn["underexp"];
        thresh = fn["thresh"];
    }

protected:
    int thresh, underexp, overexp;
    String name;
};

Ptr<GhostbusterPredict> createGhostbusterPredict(int thresh, int underexp, int overexp)
{
    return new GhostbusterPredictImpl(thresh, underexp, overexp);
}

class GhostbusterBitmapImpl : public GhostbusterBitmap
{
public:
    GhostbusterBitmapImpl(int exclude) :
      exclude(exclude),
          name("GhostbusterBitmap")
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times, Mat response)
    {
        process(src, dst);
    }

    void process(InputArrayOfArrays src, OutputArray dst)
    {
        std::vector<Mat> images;
        src.getMatVector(images);
        checkImageDimensions(images);

        int channels = images[0].channels();
        dst.create(images[0].size(), CV_8U);

        Mat res = Mat::zeros(images[0].size(), CV_8U);
        
        Ptr<AlignMTB> MTB = createAlignMTB();
        MTB->setExcludeRange(exclude);

        for(size_t i = 0; i < images.size(); i++) {
            Mat gray;
            if(channels == 1) {
                gray = images[i];
            } else {
                cvtColor(images[i], gray, COLOR_RGB2GRAY);
            }

            Mat tb, eb;
            MTB->computeBitmaps(gray, tb, eb);
            tb &= eb & 1;
            res += tb;                
        }
        res = (res > 0) & (res < images.size());
        res.copyTo(dst.getMat());
    }

    int getExclude() {return exclude;}
    void setExclude(int value) {exclude = value;}

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "exclude" << exclude;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        exclude = fn["exclude"];
    }

protected:
    int exclude;
    String name;
};

Ptr<GhostbusterBitmap> createGhostbusterBitmap(int exclude)
{
    return new GhostbusterBitmapImpl(exclude);
}

}


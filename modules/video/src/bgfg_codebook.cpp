/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

/*//The class implements the CodeBook algorithm from:
//"Real-time foreground–background segmentation using codebook model"
//Kyungnam Kima, Thanarat H. Chalidabhongseb, David Harwooda, Larry Davis
//http://www.umiacs.umd.edu/~knkim/paper/Kim-RTI2005-FinalPublished.pdf
//
//
//Example usage with as cpp class
// BackgroundSubtractorCodeBook bg_model;
//For each new image the model is updates using:
// bg_model(img, fgmask);
//
//Date: 9-May-2014, Version:1.0
///////////*/

#include "precomp.hpp"

using std::vector;

namespace cv
{

/*
 The class implements the CodeBook algorithm from:

 "Real-time foreground–background segmentation using codebook model"
 Kyungnam Kima, Thanarat H. Chalidabhongseb, David Harwooda, Larry Davis
 http://www.umiacs.umd.edu/~knkim/paper/Kim-RTI2005-FinalPublished.pdf

*/

// default parameters of codebook background detection algorithm
static const int defaultHistory = 100; // Initial learning code book
static const int defaultMaxCodeWordAge = defaultHistory/2;
static const int defaultMaxCodeWordAgeInCache = defaultHistory/2;
static const double defaultAlpha = 0.75; // [0.4 - 0.7]
static const double defaultBeta = 1.3; // [1.1 - 1.5]
static const double defaultClusteringMinColorDistance = 10;
static const double defaultDetectionMinColorDistance = 1.6*defaultClusteringMinColorDistance;
static const int defaultMinAddTime = defaultHistory;
static const int defaultMaxDeleteTime = defaultHistory;

class CodeWord {
public:
    Vec3d color;

    double minBrightness;
    double maxBrightness;

    int frequency;
    int maximumNegativeRunLength;
    int creationTime;
    int lastUpdate;
};

class CodeBook {
public:
    CodeBook(Size _size, int _maxCodeWordAge, double _alpha, double _beta, double _clusteringMinColorDistance, double _detectionMinColorDistance) {
        size = _size;
        maxCodeWordAge = _maxCodeWordAge;
        alpha = _alpha;
        beta = _beta;
        clusteringMinColorDistance = _clusteringMinColorDistance;
        detectionMinColorDistance = _detectionMinColorDistance;

        codeBook = new vector<CodeWord>[size.area()];
    }

    ~CodeBook() {
        delete [] codeBook;
    }

    vector<CodeWord>& getPixelCodeBook(int i) {
        return codeBook[i];
    }

    bool add (int i, const Vec3b& pixel, int T) {
        vector<CodeWord>& pixelCodeBook = codeBook[i];

        bool found = false;
        for (size_t k=0;k<pixelCodeBook.size();k++) {
            CodeWord& codeWord = pixelCodeBook[k];

            if (update(codeWord, pixel, T, clusteringMinColorDistance)) {
                found = true;
                break;
            }
        }

        if (!found) {
            CodeWord codeWord;

            double pixelBrightness = norm(pixel);

            codeWord.color = pixel;
            codeWord.frequency = 1;
            codeWord.minBrightness = pixelBrightness;
            codeWord.maxBrightness = pixelBrightness;
            codeWord.maximumNegativeRunLength = T - 1;
            codeWord.lastUpdate = T;
            codeWord.creationTime = T;

            pixelCodeBook.push_back(codeWord);
        }

        return found;
    }

    bool contains (int i, const Vec3b& pixel, int T) {
        vector<CodeWord>& pixelCodeBook = codeBook[i];

        for (size_t k=0;k<pixelCodeBook.size();k++) {
            CodeWord &codeWord = pixelCodeBook[k];

            if (update(codeWord, pixel, T, detectionMinColorDistance)){
                return true;
            }
        }

        return false;
    }

    void wrapAroundMNRL(int T) {
        for(int i=0; i < size.area(); i++ ) {
            vector<CodeWord> &pixelCodeBook = codeBook[i];

            for (size_t k = 0; k < pixelCodeBook.size(); k++) {
                CodeWord &codeWord = pixelCodeBook[k];

                codeWord.maximumNegativeRunLength = std::max(codeWord.maximumNegativeRunLength, T - codeWord.lastUpdate + codeWord.creationTime - 1);
            }
        }
    }

    void cleanStale(int T) {
        for(int i=0; i < size.area(); i++ ) {
            vector<CodeWord> &pixelCodeBook = codeBook[i];

            vector<CodeWord> clean;
            for (size_t k = 0; k < pixelCodeBook.size(); k++) {
                CodeWord &codeWord = pixelCodeBook[k];

                if (std::max(T-codeWord.lastUpdate,codeWord.maximumNegativeRunLength) <= maxCodeWordAge) {
                    clean.push_back(codeWord);
                }
            }

            pixelCodeBook = clean;
        }
    }


private:
    bool update (CodeWord& codeWord, const Vec3b& pixel, int T, double threshold) {
        double pixelBrightness = norm(pixel);

        double codeWordBrightness = norm(codeWord.color);
        double dotProduct = codeWord.color.dot(pixel);

        double colorDistance = std::sqrt(pixelBrightness*pixelBrightness - dotProduct*dotProduct/(codeWordBrightness*codeWordBrightness));

        double lowBrightness = codeWord.maxBrightness*alpha;
        double highBrightness = std::min(codeWord.maxBrightness*beta, codeWord.minBrightness/alpha);

        if (colorDistance <= threshold &&
                pixelBrightness >= lowBrightness &&
                pixelBrightness <= highBrightness) {
            for(int c=0;c<pixel.channels;c++){
                codeWord.color[c] = (codeWord.frequency*codeWord.color[c] + pixel[c])/(codeWord.frequency+1);
            }
            codeWord.frequency++;
            codeWord.minBrightness = std::min(codeWord.minBrightness, pixelBrightness);
            codeWord.maxBrightness = std::max(codeWord.maxBrightness, pixelBrightness);
            codeWord.maximumNegativeRunLength = std::max(codeWord.maximumNegativeRunLength, T - codeWord.lastUpdate);
            codeWord.lastUpdate = T;

            return true;
        }

        return false;
    }

    Size size;

    double detectionMinColorDistance;
    double clusteringMinColorDistance;

    double alpha;
    double beta;

    int maxCodeWordAge;

    vector<CodeWord>* codeBook;
};


class BackgroundSubtractorCodeBookImpl : public BackgroundSubtractorCodeBook
{
public:
    //! the default constructor
    BackgroundSubtractorCodeBookImpl()
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        history = defaultHistory;

        maxCodeWordAge = defaultMaxCodeWordAge;
        maxCodeWordAgeInCache = defaultMaxCodeWordAgeInCache;

        alpha = defaultAlpha;
        beta = defaultBeta;

        clusteringMinColorDistance = defaultClusteringMinColorDistance;
        detectionMinColorDistance = defaultDetectionMinColorDistance;

        minAddTime = defaultMinAddTime;
        maxDeleteTime = defaultMaxDeleteTime;

        name_ = "BackgroundSubtractor.CodeBook";
    }
    //! the full constructor that takes the length of the history,
    BackgroundSubtractorCodeBookImpl(int _history, double _alpha, double _beta, double _minColorDistance)
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        history = _history > 0 ? _history : defaultHistory;

        maxCodeWordAge = history/2;
        maxCodeWordAgeInCache = history/2;

        alpha = _alpha;
        beta = _beta;

        clusteringMinColorDistance = _minColorDistance;
        detectionMinColorDistance = 1.6*clusteringMinColorDistance;

        minAddTime = history;
        maxDeleteTime = history;

        name_ = "BackgroundSubtractor.CodeBook";
    }
    //! the destructor
    ~BackgroundSubtractorCodeBookImpl() {}
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate=-1);

    virtual void getBackgroundImage(OutputArray) const
    {
        CV_Error( Error::StsNotImplemented, "" );
    }

    //! re-initiaization method
    void initialize(Size _frameSize, int _frameType)
    {
        frameSize = _frameSize;
        frameType = _frameType;
        nframes = 0;

        int nchannels = CV_MAT_CN(frameType);
        CV_Assert( nchannels <= CV_CN_MAX );

        codeBook = makePtr<CodeBook>(frameSize, maxCodeWordAge, alpha, beta, clusteringMinColorDistance, detectionMinColorDistance);

        cache = makePtr<CodeBook>(frameSize, maxCodeWordAgeInCache, alpha, beta, clusteringMinColorDistance, detectionMinColorDistance);
    }

    virtual int getHistory() const { return history; }
    virtual void setHistory(int _nframes) { history = _nframes; }

    virtual double getAlpha() const { return alpha;}
    virtual void setAlpha(double _alpha) { alpha = _alpha; }

    virtual double getBeta() const { return beta; }
    virtual void setBeta(double _beta) { beta = _beta; }

    virtual double getClusteringMinColorDistance() const { return clusteringMinColorDistance; }
    virtual void setClusteringMinColorDistance(double threshold) { clusteringMinColorDistance = threshold; }

    virtual double getDetectionMinColorDistance() const { return detectionMinColorDistance; }
    virtual void setDetectionMinColorDistance(double threshold) { detectionMinColorDistance= threshold; }

    virtual int getMaxCodeWordAge() const { return maxCodeWordAge; }
    virtual void setMaxCodeWordAge(int age) { maxCodeWordAge = age; }

    virtual int getMaxCodeWordAgeInCache() const { return maxCodeWordAgeInCache; }
    virtual void setMaxCodeWordAgeInCache(int age) { maxCodeWordAgeInCache = age; }

    virtual int getMaxDeleteTime() const { return maxDeleteTime; }
    virtual void setMaxDeleteTime(int time) { maxDeleteTime = time; }

    virtual int getMinAddTime() const { return minAddTime; }
    virtual void setMinAddTime(int time) { minAddTime = time; }

    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
        << "history" << history
        << "detectionMinColorDistance" << detectionMinColorDistance
        << "clusteringMinColorDistance" << clusteringMinColorDistance
        << "alpha" << alpha
        << "beta" << beta
        << "maxCodeWordAge" << maxCodeWordAge
        << "maxCodeWordAgeInCache" << maxCodeWordAgeInCache
        << "maxDeleteTime" << maxDeleteTime
        << "minAddTime" << minAddTime;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        history = (int)fn["history"];
        detectionMinColorDistance = (double)fn["detectionMinColorDistance"];
        clusteringMinColorDistance = (double)fn["clusteringMinColorDistance"];
        alpha = (double)fn["alpha"];
        beta = (double)fn["beta"];
        maxCodeWordAge = (int)fn["maxCodeWordAge"];
        maxCodeWordAgeInCache = (int)fn["maxCodeWordAgeInCache"];
        maxDeleteTime = (int)fn["maxDeleteTime"];
        minAddTime = (int)fn["minAddTime"];
    }

protected:
    Size frameSize;
    int frameType;

    double detectionMinColorDistance;
    double clusteringMinColorDistance;

    double alpha;
    double beta;

    int maxCodeWordAge;
    int maxCodeWordAgeInCache;

    int maxDeleteTime;
    int minAddTime;

    Ptr<CodeBook> codeBook;
    Ptr<CodeBook> cache;

    int nframes;
    int history;

    String name_;
};

void BackgroundSubtractorCodeBookImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate) {
    if(_image.type() != CV_8UC3 )
        CV_Error( Error::StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in BackgroundSubtractorCodeBook" );

    bool needToInitialize = nframes == 0 || learningRate >= 1 || _image.size() != frameSize || _image.type() != frameType;

    if( needToInitialize )
        initialize(_image.size(), _image.type());

    Mat image = _image.getMat();
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    bool learningPhase = nframes <= history;

    for(int y = 0,i=0; y < frameSize.height; y++ )
    {
        const Vec3b* pixels = image.ptr<Vec3b>(y);
        uchar* mask = fgmask.ptr<uchar>(y);

        for(int x = 0; x < frameSize.width; x++, i++)
        {
            const Vec3b& pixel = pixels[x];

            mask[x] = 255;

            bool found = false;

            if (learningPhase) {
                found = codeBook->add(i, pixel, nframes);
            } else {
                found = codeBook->contains(i, pixel, nframes);

                if (!found) {
                    cache->add(i, pixel, nframes - history);
                }
            }

            if (found) {
                mask[x] = 0;
            }
        }
    }


    if (nframes == history) {
        codeBook->wrapAroundMNRL(nframes);
        codeBook->cleanStale(nframes);
    } else if (nframes > history) {
        cache->cleanStale(nframes-history);

        for(int i=0; i < frameSize.area(); i++ ) {
            vector<CodeWord>& pixelCodeBook = codeBook->getPixelCodeBook(i);
            vector<CodeWord>& cachePixelCodeBook = cache->getPixelCodeBook(i);

            {
                vector<CodeWord> clean;
                for (size_t k = 0; k < cachePixelCodeBook.size(); k++) {
                    CodeWord &codeWord = cachePixelCodeBook[k];

                    if (nframes - history - codeWord.creationTime > minAddTime) {
                        codeWord.creationTime += history;
                        codeWord.lastUpdate += history;
                        pixelCodeBook.push_back(codeWord);
                    } else {
                        clean.push_back(codeWord);
                    }
                }

                cachePixelCodeBook = clean;
            }

            {
                vector<CodeWord> clean;
                for (size_t k = 0; k < pixelCodeBook.size(); k++) {
                    const CodeWord &codeWord = pixelCodeBook[k];

                    if (nframes - codeWord.lastUpdate <= maxDeleteTime) {
                        clean.push_back(codeWord);
                    }
                }

                pixelCodeBook = clean;
            }
        }
    }
}

Ptr<BackgroundSubtractorCodeBook> createBackgroundSubtractorCodeBook(int _history, double _alpha, double _beta, double _colorThreshold)
{
    return makePtr<BackgroundSubtractorCodeBookImpl>(_history, _alpha, _beta, _colorThreshold);
}

}

/* End of file. */

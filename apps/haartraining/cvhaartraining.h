/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

/*
 * cvhaartraining.h
 *
 * haar training functions
 */

#ifndef _CVHAARTRAINING_H_
#define _CVHAARTRAINING_H_

class IOutput;
struct CvRect;
struct CvSize;
struct CvMat;

/*
 * cvCreateTrainingSamples
 *
 * Create training samples applying random distortions to sample image and
 * store them in .vec file
 *
 * filename        - .vec file name
 * imgfilename     - sample image file name
 * bgcolor         - background color for sample image
 * bgthreshold     - background color threshold. Pixels those colors are in range
 *   [bgcolor-bgthreshold, bgcolor+bgthreshold] are considered as transparent
 * bgfilename      - background description file name. If not NULL samples
 *   will be put on arbitrary background
 * count           - desired number of samples
 * invert          - if not 0 sample foreground pixels will be inverted
 *   if invert == CV_RANDOM_INVERT then samples will be inverted randomly
 * maxintensitydev - desired max intensity deviation of foreground samples pixels
 * maxxangle       - max rotation angles
 * maxyangle
 * maxzangle
 * showsamples     - if not 0 samples will be shown
 * winwidth        - desired samples width
 * winheight       - desired samples height
 */
#define CV_RANDOM_INVERT 0x7FFFFFFF

void cvCreateTrainingSamples(const char* filename,
                              const char* imgfilename, int bgcolor, int bgthreshold,
                              const char* bgfilename, int count,
                              int invert = 0, int maxintensitydev = 40,
                              double maxxangle = 1.1,
                              double maxyangle = 1.1,
                              double maxzangle = 0.5,
                              bool showsamples = false,
                              int winwidth = 24, int winheight = 24 );

void cvCreatePngTrainingSet(const char* imgfilename, int bgcolor, int bgthreshold,
                            const char* bgfilename, int count,
                            int invert, int maxintensitydev,
                            double maxxangle, double maxyangle, double maxzangle,
                            int winwidth, int winheight,
                            IOutput *writer );

void cvCreateTestSamples(const char* imgfilename, int bgcolor, int bgthreshold,
                          const char* bgfilename, int count,
                          int invert, int maxintensitydev,
                          double maxxangle, double maxyangle, double maxzangle,
                          int showsamples,
                          int winwidth, int winheight,
                          IOutput* writer);

/*
 * cvCreateTrainingSamplesFromInfo
 *
 * Create training samples from a set of marked up images and store them into .vec file
 * infoname    - file in which marked up image descriptions are stored
 * num         - desired number of samples
 * showsamples - if not 0 samples will be shown
 * winwidth    - sample width
 * winheight   - sample height
 *
 * Return number of successfully created samples
 */
int cvCreateTrainingSamplesFromInfo( const char* infoname, const char* vecfilename,
                                     int num,
                                     int showsamples,
                                     int winwidth, int winheight );

/*
 * cvShowVecSamples
 *
 * Shows samples stored in .vec file
 *
 * filename
 *   .vec file name
 * winwidth
 *   sample width
 * winheight
 *   sample height
 * scale
 *   the scale each sample is adjusted to
 */
void cvShowVecSamples( const char* filename, int winwidth, int winheight, double scale );


/*
 * cvCreateCascadeClassifier
 *
 * Create cascade classifier
 * dirname          - directory name in which cascade classifier will be created.
 *   It must exist and contain subdirectories 0, 1, 2, ... (nstages-1).
 * vecfilename      - name of .vec file with object's images
 * bgfilename       - name of background description file
 * bg_vecfile       - true if bgfilename represents a vec file with discrete negatives
 * npos             - number of positive samples used in training of each stage
 * nneg             - number of negative samples used in training of each stage
 * nstages          - number of stages
 * numprecalculated - number of features being precalculated. Each precalculated feature
 *   requires (number_of_samples*(sizeof( float ) + sizeof( short ))) bytes of memory
 * numsplits        - number of binary splits in each weak classifier
 *   1 - stumps, 2 and more - trees.
 * minhitrate       - desired min hit rate of each stage
 * maxfalsealarm    - desired max false alarm of each stage
 * weightfraction   - weight trimming parameter
 * mode             - 0 - BASIC = Viola
 *                    1 - CORE  = All upright
 *                    2 - ALL   = All features
 * symmetric        - if not 0 vertical symmetry is assumed
 * equalweights     - if not 0 initial weights of all samples will be equal
 * winwidth         - sample width
 * winheight        - sample height
 * boosttype        - type of applied boosting algorithm
 *   0 - Discrete AdaBoost
 *   1 - Real AdaBoost
 *   2 - LogitBoost
 *   3 - Gentle AdaBoost
 * stumperror       - type of used error if Discrete AdaBoost algorithm is applied
 *   0 - misclassification error
 *   1 - gini error
 *   2 - entropy error
 */
void cvCreateCascadeClassifier( const char* dirname,
                                const char* vecfilename,
                                const char* bgfilename,
                                int npos, int nneg, int nstages,
                                int numprecalculated,
                                int numsplits,
                                float minhitrate = 0.995F, float maxfalsealarm = 0.5F,
                                float weightfraction = 0.95F,
                                int mode = 0, int symmetric = 1,
                                int equalweights = 1,
                                int winwidth = 24, int winheight = 24,
                                int boosttype = 3, int stumperror = 0 );

void cvCreateTreeCascadeClassifier( const char* dirname,
                                    const char* vecfilename,
                                    const char* bgfilename,
                                    int npos, int nneg, int nstages,
                                    int numprecalculated,
                                    int numsplits,
                                    float minhitrate, float maxfalsealarm,
                                    float weightfraction,
                                    int mode, int symmetric,
                                    int equalweights,
                                    int winwidth, int winheight,
                                    int boosttype, int stumperror,
                                    int maxtreesplits, int minpos, bool bg_vecfile = false );


class DatasetGenerator
{
public:
    DatasetGenerator( IOutput* _writer );
    void create( const char* imgfilename, int bgcolor, int bgthreshold,
                 const char* bgfilename, int count,
                 int invert, int maxintensitydev,
                 double maxxangle, double maxyangle, double maxzangle,
                 bool showsamples,
                 int winwidth, int winheight);
    virtual ~DatasetGenerator();
private:
    virtual void showSamples( bool* showSamples, CvMat* img ) const;

    CvRect getObjectPosition( const CvSize& bgImgSize,
                              const CvSize& imgSize,
                              const CvSize& sampleSize ) const;
    virtual CvSize scaleObjectSize(const CvSize& bgImgSize,
                                   const CvSize& imgSize ,
                                   const CvSize& sampleSize) const =0 ;
private:
    IOutput* writer;
};

/* Provides the functionality of test set generating */
class JpgDatasetGenerator: public DatasetGenerator
{
public:
    JpgDatasetGenerator(const char* filename);
private:
    CvSize scaleObjectSize(const CvSize& bgImgSize,
                           const CvSize& ,
                           const CvSize& sampleSize) const;
};

class PngDatasetGenerator: public DatasetGenerator
{
public:
    PngDatasetGenerator(const char *filename);
private:
    CvSize scaleObjectSize(const CvSize& bgImgSize,
                           const CvSize& imgSize ,
                           const CvSize& ) const;
};

#endif /* _CVHAARTRAINING_H_ */

/*
 * BackgroundSubtractorGBH_test.cpp
 *
 *  Created on: Jun 14, 2012
 *      Author: andrewgodbehere
 */

#include "test_precomp.hpp"

using namespace cv;

class CV_BackgroundSubtractorTest : public cvtest::BaseTest
{
public:
    CV_BackgroundSubtractorTest();
protected:
    void run(int);
};

CV_BackgroundSubtractorTest::CV_BackgroundSubtractorTest()
{
}

/**
 * This test checks the following:
 * (i) BackgroundSubtractorGMG can operate with matrices of various types and sizes
 * (ii) Training mode returns empty fgmask
 * (iii) End of training mode, and anomalous frame yields every pixel detected as FG
 */
void CV_BackgroundSubtractorTest::run(int)
{
    int code = cvtest::TS::OK;
    RNG& rng = ts->get_rng();
    int type = ((unsigned int)rng)%7;  //!< pick a random type, 0 - 6, defined in types_c.h
    int channels = 1 + ((unsigned int)rng)%4;  //!< random number of channels from 1 to 4.
    int channelsAndType = CV_MAKETYPE(type,channels);
    int width = 2 + ((unsigned int)rng)%98; //!< Mat will be 2 to 100 in width and height
    int height = 2 + ((unsigned int)rng)%98;

    Ptr<BackgroundSubtractorGMG> fgbg = createBackgroundSubtractorGMG();
    Mat fgmask;

    if (!fgbg)
        CV_Error(Error::StsError,"Failed to create Algorithm\n");

    /**
     * Set a few parameters
     */
    fgbg->setSmoothingRadius(7);
    fgbg->setDecisionThreshold(0.7);
    fgbg->setNumFrames(120);

    /**
     * Generate bounds for the values in the matrix for each type
     */
    double maxd = 0, mind = 0;

    /**
     * Max value for simulated images picked randomly in upper half of type range
     * Min value for simulated images picked randomly in lower half of type range
     */
    if (type == CV_8U)
    {
        uchar half = UCHAR_MAX/2;
        maxd = (unsigned char)rng.uniform(half+32, UCHAR_MAX);
        mind = (unsigned char)rng.uniform(0, half-32);
    }
    else if (type == CV_8S)
    {
        maxd = (char)rng.uniform(32, CHAR_MAX);
        mind = (char)rng.uniform(CHAR_MIN, -32);
    }
    else if (type == CV_16U)
    {
        ushort half = USHRT_MAX/2;
        maxd = (unsigned int)rng.uniform(half+32, USHRT_MAX);
        mind = (unsigned int)rng.uniform(0, half-32);
    }
    else if (type == CV_16S)
    {
        maxd = rng.uniform(32, SHRT_MAX);
        mind = rng.uniform(SHRT_MIN, -32);
    }
    else if (type == CV_32S)
    {
        maxd = rng.uniform(32, INT_MAX);
        mind = rng.uniform(INT_MIN, -32);
    }
    else if (type == CV_32F)
    {
        maxd = rng.uniform(32.0f, FLT_MAX);
        mind = rng.uniform(-FLT_MAX, -32.0f);
    }
    else if (type == CV_64F)
    {
        maxd = rng.uniform(32.0, DBL_MAX);
        mind = rng.uniform(-DBL_MAX, -32.0);
    }

    fgbg->setMinVal(mind);
    fgbg->setMaxVal(maxd);

    Mat simImage = Mat::zeros(height, width, channelsAndType);
    int numLearningFrames = 120;
    for (int i = 0; i < numLearningFrames; ++i)
    {
        /**
         * Genrate simulated "image" for any type. Values always confined to upper half of range.
         */
        rng.fill(simImage, RNG::UNIFORM, (mind + maxd)*0.5, maxd);

        /**
         * Feed simulated images into background subtractor
         */
        fgbg->apply(simImage,fgmask);
        Mat fullbg = Mat::zeros(simImage.rows, simImage.cols, CV_8U);

        //! fgmask should be entirely background during training
        code = cvtest::cmpEps2( ts, fgmask, fullbg, 0, false, "The training foreground mask" );
        if (code < 0)
            ts->set_failed_test_info( code );
    }
    //! generate last image, distinct from training images
    rng.fill(simImage, RNG::UNIFORM, mind, maxd);

    fgbg->apply(simImage,fgmask);
    //! now fgmask should be entirely foreground
    Mat fullfg = 255*Mat::ones(simImage.rows, simImage.cols, CV_8U);
    code = cvtest::cmpEps2( ts, fgmask, fullfg, 255, false, "The final foreground mask" );
    if (code < 0)
    {
        ts->set_failed_test_info( code );
    }

}

TEST(VIDEO_BGSUBGMG, accuracy) { CV_BackgroundSubtractorTest test; test.safe_run(); }

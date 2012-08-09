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

    Ptr<BackgroundSubtractorGMG> fgbg =
            Algorithm::create<BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
    Mat fgmask;

    if (fgbg == NULL)
        CV_Error(CV_StsError,"Failed to create Algorithm\n");

    /**
     * Set a few parameters
     */
    fgbg->set("smoothingRadius",7);
    fgbg->set("decisionThreshold",0.7);
    fgbg->set("initializationFrames",120);

    /**
     * Generate bounds for the values in the matrix for each type
     */
    uchar maxuc = 0, minuc = 0;
    char maxc = 0, minc = 0;
    unsigned int maxui = 0, minui = 0;
    int maxi=0, mini = 0;
    long int maxli = 0, minli = 0;
    float maxf = 0, minf = 0;
    double maxd = 0, mind = 0;

    /**
     * Max value for simulated images picked randomly in upper half of type range
     * Min value for simulated images picked randomly in lower half of type range
     */
    if (type == CV_8U)
    {
        uchar half = UCHAR_MAX/2;
        maxuc = (unsigned char)rng.uniform(half+32, UCHAR_MAX);
        minuc = (unsigned char)rng.uniform(0, half-32);
    }
    else if (type == CV_8S)
    {
        maxc = (char)rng.uniform(32, CHAR_MAX);
        minc = (char)rng.uniform(CHAR_MIN, -32);
    }
    else if (type == CV_16U)
    {
        ushort half = USHRT_MAX/2;
        maxui = (unsigned int)rng.uniform(half+32, USHRT_MAX);
        minui = (unsigned int)rng.uniform(0, half-32);
    }
    else if (type == CV_16S)
    {
        maxi = rng.uniform(32, SHRT_MAX);
        mini = rng.uniform(SHRT_MIN, -32);
    }
    else if (type == CV_32S)
    {
        maxli = rng.uniform(32, INT_MAX);
        minli = rng.uniform(INT_MIN, -32);
    }
    else if (type == CV_32F)
    {
        maxf = rng.uniform(32.0f, FLT_MAX);
        minf = rng.uniform(-FLT_MAX, -32.0f);
    }
    else if (type == CV_64F)
    {
        maxd = rng.uniform(32.0, DBL_MAX);
        mind = rng.uniform(-DBL_MAX, -32.0);
    }

    Mat simImage = Mat::zeros(height, width, channelsAndType);
    const unsigned int numLearningFrames = 120;
    for (unsigned int i = 0; i < numLearningFrames; ++i)
    {
        /**
         * Genrate simulated "image" for any type. Values always confined to upper half of range.
         */
        if (type == CV_8U)
        {
            rng.fill(simImage,RNG::UNIFORM,(unsigned char)(minuc/2+maxuc/2),maxuc);
            if (i == 0)
                fgbg->initialize(simImage.size(),minuc,maxuc);
        }
        else if (type == CV_8S)
        {
            rng.fill(simImage,RNG::UNIFORM,(char)(minc/2+maxc/2),maxc);
            if (i==0)
                fgbg->initialize(simImage.size(),minc,maxc);
        }
        else if (type == CV_16U)
        {
            rng.fill(simImage,RNG::UNIFORM,(unsigned int)(minui/2+maxui/2),maxui);
            if (i==0)
                fgbg->initialize(simImage.size(),minui,maxui);
        }
        else if (type == CV_16S)
        {
            rng.fill(simImage,RNG::UNIFORM,(int)(mini/2+maxi/2),maxi);
            if (i==0)
                fgbg->initialize(simImage.size(),mini,maxi);
        }
        else if (type == CV_32F)
        {
            rng.fill(simImage,RNG::UNIFORM,(float)(minf/2.0+maxf/2.0),maxf);
            if (i==0)
                fgbg->initialize(simImage.size(),minf,maxf);
        }
        else if (type == CV_32S)
        {
            rng.fill(simImage,RNG::UNIFORM,(long int)(minli/2+maxli/2),maxli);
            if (i==0)
                fgbg->initialize(simImage.size(),minli,maxli);
        }
        else if (type == CV_64F)
        {
            rng.fill(simImage,RNG::UNIFORM,(double)(mind/2.0+maxd/2.0),maxd);
            if (i==0)
                fgbg->initialize(simImage.size(),mind,maxd);
        }

        /**
         * Feed simulated images into background subtractor
         */
        (*fgbg)(simImage,fgmask);
        Mat fullbg = Mat::zeros(simImage.rows, simImage.cols, CV_8U);

        //! fgmask should be entirely background during training
        code = cvtest::cmpEps2( ts, fgmask, fullbg, 0, false, "The training foreground mask" );
        if (code < 0)
            ts->set_failed_test_info( code );
    }
    //! generate last image, distinct from training images
    if (type == CV_8U)
        rng.fill(simImage,RNG::UNIFORM,minuc,minuc);
    else if (type == CV_8S)
        rng.fill(simImage,RNG::UNIFORM,minc,minc);
    else if (type == CV_16U)
        rng.fill(simImage,RNG::UNIFORM,minui,minui);
    else if (type == CV_16S)
        rng.fill(simImage,RNG::UNIFORM,mini,mini);
    else if (type == CV_32F)
        rng.fill(simImage,RNG::UNIFORM,minf,minf);
    else if (type == CV_32S)
        rng.fill(simImage,RNG::UNIFORM,minli,minli);
    else if (type == CV_64F)
        rng.fill(simImage,RNG::UNIFORM,mind,mind);

    (*fgbg)(simImage,fgmask);
    //! now fgmask should be entirely foreground
    Mat fullfg = 255*Mat::ones(simImage.rows, simImage.cols, CV_8U);
    code = cvtest::cmpEps2( ts, fgmask, fullfg, 255, false, "The final foreground mask" );
    if (code < 0)
    {
        ts->set_failed_test_info( code );
    }

}

TEST(VIDEO_BGSUBGMG, accuracy) { CV_BackgroundSubtractorTest test; test.safe_run(); }

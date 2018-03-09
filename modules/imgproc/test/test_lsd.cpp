// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

const Size img_size(640, 480);
const int LSD_TEST_SEED = 0x134679;
const int EPOCHS = 20;

class LSDBase : public testing::Test
{
public:
    LSDBase() { }

protected:
    Mat test_image;
    vector<Vec4f> lines;
    RNG rng;
    int passedtests;

    void GenerateWhiteNoise(Mat& image);
    void GenerateConstColor(Mat& image);
    void GenerateLines(Mat& image, const unsigned int numLines);
    void GenerateRotatedRect(Mat& image);
    virtual void SetUp();
};

class Imgproc_LSD_ADV: public LSDBase
{
public:
    Imgproc_LSD_ADV() { }
protected:

};

class Imgproc_LSD_STD: public LSDBase
{
public:
    Imgproc_LSD_STD() { }
protected:

};

class Imgproc_LSD_NONE: public LSDBase
{
public:
    Imgproc_LSD_NONE() { }
protected:

};

void LSDBase::GenerateWhiteNoise(Mat& image)
{
    image = Mat(img_size, CV_8UC1);
    rng.fill(image, RNG::UNIFORM, 0, 256);
}

void LSDBase::GenerateConstColor(Mat& image)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 256)));
}

void LSDBase::GenerateLines(Mat& image, const unsigned int numLines)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 128)));

    for(unsigned int i = 0; i < numLines; ++i)
    {
        int y = rng.uniform(10, img_size.width - 10);
        Point p1(y, 10);
        Point p2(y, img_size.height - 10);
        line(image, p1, p2, Scalar(255), 3);
    }
}

void LSDBase::GenerateRotatedRect(Mat& image)
{
    image = Mat::zeros(img_size, CV_8UC1);

    Point center(rng.uniform(img_size.width/4, img_size.width*3/4),
                 rng.uniform(img_size.height/4, img_size.height*3/4));
    Size rect_size(rng.uniform(img_size.width/8, img_size.width/6),
                   rng.uniform(img_size.height/8, img_size.height/6));
    float angle = rng.uniform(0.f, 360.f);

    Point2f vertices[4];

    RotatedRect rRect = RotatedRect(center, rect_size, angle);

    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255), 3);
    }
}

void LSDBase::SetUp()
{
    lines.clear();
    test_image = Mat();
    rng = RNG(LSD_TEST_SEED);
    passedtests = 0;
}


TEST_F(Imgproc_LSD_ADV, whiteNoise)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateWhiteNoise(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
        detector->detect(test_image, lines);

        if(40u >= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_ADV, constColor)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateConstColor(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
        detector->detect(test_image, lines);

        if(0u == lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_ADV, lines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateLines(test_image, numOfLines);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
        detector->detect(test_image, lines);

        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_ADV, rotatedRect)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateRotatedRect(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
        detector->detect(test_image, lines);

        if(2u <= lines.size())  ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_STD, whiteNoise)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateWhiteNoise(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
        detector->detect(test_image, lines);

        if(50u >= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_STD, constColor)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateConstColor(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
        detector->detect(test_image, lines);

        if(0u == lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_STD, lines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateLines(test_image, numOfLines);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
        detector->detect(test_image, lines);

        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_STD, rotatedRect)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateRotatedRect(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
        detector->detect(test_image, lines);

        if(4u <= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_NONE, whiteNoise)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateWhiteNoise(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_STD);
        detector->detect(test_image, lines);

        if(50u >= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_NONE, constColor)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateConstColor(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        detector->detect(test_image, lines);

        if(0u == lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_NONE, lines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateLines(test_image, numOfLines);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        detector->detect(test_image, lines);

        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_NONE, rotatedRect)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateRotatedRect(test_image);
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        detector->detect(test_image, lines);

        if(8u <= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

}} // namespace

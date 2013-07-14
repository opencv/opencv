#include "test_precomp.hpp"

#include <vector>

using namespace cv;
using namespace std;

const Size img_size(640, 480);

class LSDBase : public testing::Test
{
public:
    LSDBase() {};

protected:
    Mat test_image;
    vector<Vec4i> lines;

    void GenerateWhiteNoise(Mat& image);
    void GenerateConstColor(Mat& image);
    void GenerateLines(Mat& image, const unsigned int numLines);
    void GenerateRotatedRect(Mat& image);
    virtual void SetUp();
};

class LSD_ADV: public LSDBase
{
public:
    LSD_ADV() {};
protected:

};

class LSD_STD: public LSDBase
{
public:
    LSD_STD() {};
protected:

};

class LSD_NONE: public LSDBase
{
public:
    LSD_NONE() {};
protected:

};

void LSDBase::GenerateWhiteNoise(Mat& image)
{
    image = Mat(img_size, CV_8UC1);
    RNG rng(getTickCount());
    rng.fill(image, RNG::UNIFORM, 0, 256);
}

void LSDBase::GenerateConstColor(Mat& image)
{
    RNG rng(getTickCount());
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 256)));
}

void LSDBase::GenerateLines(Mat& image, const unsigned int numLines)
{
    RNG rng(getTickCount());
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 128)));

    for(unsigned int i = 0; i < numLines; ++i)
    {
        int y = rng.uniform(10, img_size.width - 10);
        Point p1(y, 10);
        Point p2(y, img_size.height - 10);
        line(image, p1, p2, Scalar(255), 1);
    }
}

void LSDBase::GenerateRotatedRect(Mat& image)
{
    RNG rng(getTickCount());
    image = Mat::zeros(img_size, CV_8UC1);

    Point center(rng.uniform(img_size.width/4, img_size.width*3/4),
                 rng.uniform(img_size.height/4, img_size.height*3/4));
    Size rect_size(rng.uniform(img_size.width/8, img_size.width/6),
                   rng.uniform(img_size.height/8, img_size.height/6));
    float angle = rng.uniform(0, 360);

    Point2f vertices[4];

    RotatedRect rRect = RotatedRect(center, rect_size, angle);

    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255));
    }
}

void LSDBase::SetUp()
{
    lines.clear();
    test_image = Mat();
}


TEST_F(LSD_ADV, whiteNoise)
{
    GenerateWhiteNoise(test_image);
    LSD detector(LSD_REFINE_ADV);
    detector.detect(test_image, lines);

    ASSERT_GE((unsigned int)(40), lines.size());
}

TEST_F(LSD_ADV, constColor)
{
    GenerateConstColor(test_image);
    LSD detector(LSD_REFINE_ADV);
    detector.detect(test_image, lines);

    ASSERT_EQ((unsigned int)(0), lines.size());
}

TEST_F(LSD_ADV, lines)
{
    const unsigned int numOfLines = 3;
    GenerateLines(test_image, numOfLines);
    LSD detector(LSD_REFINE_ADV);
    detector.detect(test_image, lines);

    ASSERT_EQ(numOfLines * 2, lines.size()); // * 2 because of Gibbs effect
}

TEST_F(LSD_ADV, rotatedRect)
{
    GenerateRotatedRect(test_image);
    LSD detector(LSD_REFINE_ADV);
    detector.detect(test_image, lines);
    ASSERT_LE((unsigned int)(4), lines.size());
}

TEST_F(LSD_STD, whiteNoise)
{
    GenerateWhiteNoise(test_image);
    LSD detector(LSD_REFINE_STD);
    detector.detect(test_image, lines);

    ASSERT_GE((unsigned int)(50), lines.size());
}

TEST_F(LSD_STD, constColor)
{
    GenerateConstColor(test_image);
    LSD detector(LSD_REFINE_STD);
    detector.detect(test_image, lines);

    ASSERT_EQ((unsigned int)(0), lines.size());
}

TEST_F(LSD_STD, lines)
{
    const unsigned int numOfLines = 3; //1
    GenerateLines(test_image, numOfLines);
    LSD detector(LSD_REFINE_STD);
    detector.detect(test_image, lines);

    ASSERT_EQ(numOfLines * 2, lines.size()); // * 2 because of Gibbs effect
}

TEST_F(LSD_STD, rotatedRect)
{
    GenerateRotatedRect(test_image);
    LSD detector(LSD_REFINE_STD);
    detector.detect(test_image, lines);
    ASSERT_EQ((unsigned int)(8), lines.size());
}

TEST_F(LSD_NONE, whiteNoise)
{
    GenerateWhiteNoise(test_image);
    LSD detector(LSD_REFINE_NONE);
    detector.detect(test_image, lines);

    ASSERT_GE((unsigned int)(50), lines.size());
}

TEST_F(LSD_NONE, constColor)
{
    GenerateConstColor(test_image);
    LSD detector(LSD_REFINE_NONE);
    detector.detect(test_image, lines);

    ASSERT_EQ((unsigned int)(0), lines.size());
}

TEST_F(LSD_NONE, lines)
{
    const unsigned int numOfLines = 3; //1
    GenerateLines(test_image, numOfLines);
    LSD detector(LSD_REFINE_NONE);
    detector.detect(test_image, lines);

    ASSERT_EQ(numOfLines * 2, lines.size()); // * 2 because of Gibbs effect
}

TEST_F(LSD_NONE, rotatedRect)
{
    GenerateRotatedRect(test_image);
    LSD detector(LSD_REFINE_NONE);
    detector.detect(test_image, lines);
    ASSERT_EQ((unsigned int)(8), lines.size());
}

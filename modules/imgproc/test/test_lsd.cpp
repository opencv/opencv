#include "test_precomp.hpp"

#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

const Size img_size(640, 480);
const int LSD_TEST_SEED = 0x134679;
const int EPOCHS = 20;
const int DISPL = 20;
const Point init_pnt(17, 13);
const int NUM_LINES_F = 10;

class LSDBase : public testing::Test
{
public:
    LSDBase() { }

protected:
    Mat test_image;
    vector<Vec4i> lines;
    RNG rng;
    int passedtests;

    void GenerateWhiteNoise(Mat& image);
    void GenerateConstColor(Mat& image);
    void GenerateLines(Mat& image, const unsigned int numLines);
    void GenerateRotatedRect(Mat& image);
    void AddLine(vector<Vec4i>& l, const Point p, const float angle_deg, const double length);
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

class Imgproc_LSD_FILTERING: public LSDBase
{
public:
    Imgproc_LSD_FILTERING() { }
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

void LSDBase::AddLine(vector<Vec4i>& l, const Point p, const float angle_deg, const double length)
{
    int dy = int(sin(angle_deg * CV_PI / 180.0) * length);
    int dx = int(cos(angle_deg * CV_PI / 180.0) * length);

    l.push_back(Vec4i(p.x,
                      p.y,
                      p.x + dx,
                      p.y + dy));
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

TEST_F(Imgproc_LSD_FILTERING, min_size)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        lines.clear();
        for (int l = 0; l < NUM_LINES_F; ++l)
            AddLine(lines, init_pnt, 0, l * DISPL); // Generate NUM_LINES_F lines

        vector<Vec4i> filtered;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        double min_length = NUM_LINES_F * DISPL / 2.0;

        detector->filterSize(lines, filtered, min_length);
        cout << filtered.size() << endl;

        if(filtered.size() == (NUM_LINES_F / 2u)) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, max_size)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        lines.clear();
        for (int l = 0; l < NUM_LINES_F; ++l)
            AddLine(lines, init_pnt, 0, l * DISPL); // Generate NUM_LINES_F lines

        vector<Vec4i> filtered;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        double max_length = NUM_LINES_F * DISPL / 2.0;
        detector->filterSize(lines, filtered, 0, max_length);

        if(filtered.size() == (NUM_LINES_F / 2u)) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, angles_retain)
{
    const int split_pi = 18; // split 180deg in 18 parts
    for (int i = 0; i < split_pi; ++i) // in ordder to select every line
    {
        lines.clear();
        for (int a = 0; a < 180; a+=10)
            AddLine(lines, init_pnt, float(a), DISPL); // Generate lines at 10deg separation

        vector<Vec4i> filtered;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);

        double min_angle = max(0.0,   i * 10.0 - 5.0);
        double max_angle = min(179.9, i * 10.0 + 5.0);
        detector->retainAngle(lines, filtered, min_angle, max_angle); // select one line

        if(filtered.size() == 1u) ++passedtests;
    }
    ASSERT_EQ(split_pi, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, angles_filterOut)
{
    const int split_pi = 18; // split 180deg in 18 parts
    for (int i = 0; i < split_pi; ++i) // in ordder to select every line
    {
        lines.clear();
        for (int a = 0; a < 180; a+=10)
            AddLine(lines, init_pnt, float(a), DISPL); // Generate lines at 10deg separation

        vector<Vec4i> filtered;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);

        double min_angle = max(0.0,   i * 10.0 - 5.0);
        double max_angle = min(179.9, i * 10.0 + 5.0);
        detector->filterOutAngle(lines, filtered, min_angle, max_angle); // select one line

        if(filtered.size() == (split_pi - 1u)) ++passedtests;
    }
    ASSERT_EQ(split_pi, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, intersection_parallel)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        lines.clear();
        AddLine(lines, init_pnt, 0, DISPL * (i + 1)); // Line A
        AddLine(lines, init_pnt, 0, DISPL * (i + 2)); // Line B, A||B

        Point P;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        if (!detector->intersection(lines[0], lines[1], P)) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, intersection_perp)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        lines.clear();
        AddLine(lines, init_pnt * i,  0.0, DISPL * (i + 1));
        AddLine(lines, init_pnt * i, 90.0, DISPL * (i + 2));

        Point P;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        detector->intersection(lines[0], lines[1], P);
        if (P == (init_pnt * i)) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(Imgproc_LSD_FILTERING, intersections)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        lines.clear();
        AddLine(lines, init_pnt * i, 0.0, DISPL);
        AddLine(lines, init_pnt, float((i + 1) * 8.0), DISPL);

        Point P;
        Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_NONE);
        bool not_parallel = detector->intersection(lines[0], lines[1], P);
        if (not_parallel && P!=Point()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

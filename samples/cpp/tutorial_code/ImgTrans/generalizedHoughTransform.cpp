/**
  @file generalizedHoughTransform.cpp
  @author Markus Heck
  @brief Detects an object, given by a template, in an image using GeneralizedHoughBallard and GeneralizedHoughGuil.
*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
    //! [generalized-hough-transform-load-and-setup]
//  load source image and grayscale template
    samples::addSamplesDataSearchSubDirectory("doc/tutorials/imgproc/generalized_hough_ballard_guil");
    Mat image = imread(samples::findFile("images/generalized_hough_mini_image.jpg"));
    Mat templ = imread(samples::findFile("images/generalized_hough_mini_template.jpg"), IMREAD_GRAYSCALE);

//  create grayscale image
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_RGB2GRAY);

//  create variable for location, scale and rotation of detected templates
    vector<Vec4f> positionBallard, positionGuil;

//  template width and height
    int w = templ.cols;
    int h = templ.rows;
    //! [generalized-hough-transform-load-and-setup]


    //! [generalized-hough-transform-setup-parameters]
//  create ballard and set options
    Ptr<GeneralizedHoughBallard> ballard = createGeneralizedHoughBallard();
    ballard->setMinDist(10);
    ballard->setLevels(360);
    ballard->setDp(2);
    ballard->setMaxBufferSize(1000);
    ballard->setVotesThreshold(40);

    ballard->setCannyLowThresh(30);
    ballard->setCannyHighThresh(110);
    ballard->setTemplate(templ);


//  create guil and set options
    Ptr<GeneralizedHoughGuil> guil = createGeneralizedHoughGuil();
    guil->setMinDist(10);
    guil->setLevels(360);
    guil->setDp(3);
    guil->setMaxBufferSize(1000);

    guil->setMinAngle(0);
    guil->setMaxAngle(360);
    guil->setAngleStep(1);
    guil->setAngleThresh(1500);

    guil->setMinScale(0.5);
    guil->setMaxScale(2.0);
    guil->setScaleStep(0.05);
    guil->setScaleThresh(50);

    guil->setPosThresh(10);

    guil->setCannyLowThresh(30);
    guil->setCannyHighThresh(110);

    guil->setTemplate(templ);
    //! [generalized-hough-transform-setup-parameters]


    //! [generalized-hough-transform-run]
//  execute ballard detection
    ballard->detect(grayImage, positionBallard);
//  execute guil detection
    guil->detect(grayImage, positionGuil);
    //! [generalized-hough-transform-run]


    //! [generalized-hough-transform-draw-results]
//  draw ballard
    for (vector<Vec4f>::iterator iter = positionBallard.begin(); iter != positionBallard.end(); ++iter) {
        RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]),
                                        Size2f(w * (*iter)[2], h * (*iter)[2]),
                                        (*iter)[3]);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 6);
    }

//  draw guil
    for (vector<Vec4f>::iterator iter = positionGuil.begin(); iter != positionGuil.end(); ++iter) {
        RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]),
                                        Size2f(w * (*iter)[2], h * (*iter)[2]),
                                        (*iter)[3]);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }

    imshow("result_img", image);
    waitKey();
    //! [generalized-hough-transform-draw-results]

    return EXIT_SUCCESS;
}

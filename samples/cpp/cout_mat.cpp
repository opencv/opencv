/*
 *
 * cvout_sample just demonstrates the serial out capabilities of cv::Mat
 *  That is, cv::Mat M(...); cout << M;  Now works.
 *
 */

#include "opencv2/core.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout
    << "\n------------------------------------------------------------------\n"
    << " This program shows the serial out capabilities of cv::Mat\n"
    << "That is, cv::Mat M(...); cout << M;  Now works.\n"
    << "Output can be formatted to OpenCV, matlab, python, numpy, csv and \n"
    << "C styles Usage:\n"
    << argv[0]
    << "\n------------------------------------------------------------------\n\n"
    << endl;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    ofstream outfile("output.txt");
    if (!outfile.is_open())
    {
        cerr << "Could not open the file for writing!" << endl;
        return -1;
    }

    Mat I = Mat::eye(4, 4, CV_64F);
    I.at<double>(1,1) = CV_PI;
    outfile << "I = \n" << I << ";" << endl << endl;

    Mat r = Mat(10, 3, CV_8UC3);
    randu(r, Scalar::all(0), Scalar::all(255));

    outfile << "r (default) = \n" << r << ";" << endl << endl;
    outfile << "r (matlab) = \n" << format(r, Formatter::FMT_MATLAB) << ";" << endl << endl;
    outfile << "r (python) = \n" << format(r, Formatter::FMT_PYTHON) << ";" << endl << endl;
    outfile << "r (numpy) = \n" << format(r, Formatter::FMT_NUMPY) << ";" << endl << endl;
    outfile << "r (csv) = \n" << format(r, Formatter::FMT_CSV) << ";" << endl << endl;
    outfile << "r (c) = \n" << format(r, Formatter::FMT_C) << ";" << endl << endl;

    Point2f p(5, 1);
    outfile << "p = " << p << ";" << endl;

    Point3f p3f(2, 6, 7);
    outfile << "p3f = " << p3f << ";" << endl;

    vector<float> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    outfile << "shortvec = " << Mat(v) << endl;

    vector<Point2f> points(20);
    for (size_t i = 0; i < points.size(); ++i)
        points[i] = Point2f((float)(i * 5), (float)(i % 7));

    outfile << "points = " << points << ";" << endl;

    outfile.close();
    cout << "Output written to output.txt" << endl;

    return 0;
}


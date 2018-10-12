#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void loadExposureSeq(String, vector<Mat>&, vector<float>&);

int main(int argc, char**argv)
{
    CommandLineParser parser( argc, argv, "{@input | | Input directory that contains images and exposure times. }" );

    //! [Load images and exposure times]
    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(parser.get<String>( "@input" ), images, times);
    //! [Load images and exposure times]

    //! [Estimate camera response]
    Mat response;
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
    calibrate->process(images, response, times);
    //! [Estimate camera response]

    //! [Make HDR image]
    Mat hdr;
    Ptr<MergeDebevec> merge_debevec = createMergeDebevec();
    merge_debevec->process(images, hdr, times, response);
    //! [Make HDR image]

    //! [Tonemap HDR image]
    Mat ldr;
    Ptr<TonemapDurand> tonemap = createTonemapDurand(2.2f);
    tonemap->process(hdr, ldr);
    //! [Tonemap HDR image]

    //! [Perform exposure fusion]
    Mat fusion;
    Ptr<MergeMertens> merge_mertens = createMergeMertens();
    merge_mertens->process(images, fusion);
    //! [Perform exposure fusion]

    //! [Write results]
    imwrite("fusion.png", fusion * 255);
    imwrite("ldr.png", ldr * 255);
    imwrite("hdr.hdr", hdr);
    //! [Write results]

    return 0;
}

void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times)
{
    path = path + "/";
    ifstream list_file((path + "list.txt").c_str());
    string name;
    float val;
    while(list_file >> name >> val) {
        Mat img = imread(path + name);
        images.push_back(img);
        times.push_back(1 / val);
    }
    list_file.close();
}

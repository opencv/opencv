// Sample of using Halide backend in OpenCV deep learning module.
// Based on caffe_googlenet.cpp.

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

static std::vector<std::string> readClassNames(const char *filename = "synset_words.txt")
{
    std::vector<std::string> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}

int main(int argc, char **argv)
{
    std::string modelTxt = "train_val.prototxt";
    std::string modelBin = "squeezenet_v1.1.caffemodel";
    std::string imageFile = (argc > 1) ? argv[1] : "space_shuttle.jpg";

    //! [Read and initialize network]
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    //! [Read and initialize network]

    //! [Check that network was read successfully]
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        std::cerr << "SqueezeNet v1.1 can be downloaded from:" << std::endl;
        std::cerr << "https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1" << std::endl;
        exit(-1);
    }
    //! [Check that network was read successfully]

    //! [Prepare blob]
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    if (img.channels() != 3)
    {
        std::cerr << "Image " << imageFile << " isn't 3-channel" << std::endl;
        exit(-1);
    }

    resize(img, img, Size(227, 227));                // SqueezeNet v1.1 predict class by 3x227x227 input image.
    Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(), false);  // Convert Mat to 4-dimensional batch.
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob);                         // Set the network input.
    //! [Set input blob]

    //! [Enable Halide backend]
    net.setPreferableBackend(DNN_BACKEND_HALIDE);    // Tell engine to use Halide where it possible.
    //! [Enable Halide backend]

    //! [Make forward pass]
    Mat prob = net.forward("prob");                  // Compute output.
    //! [Make forward pass]

    //! [Determine the best class]
    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);         // Find the best class.
    //! [Determine the best class]

    //! [Print results]
    std::vector<std::string> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    //! [Print results]

    return 0;
} //main

//
// this sample demonstrates parsing (segmenting) human body parts from an image using opencv's dnn,
// based on https://github.com/Engineering-Course/LIP_JPPNet
//
// get the pretrained model from: https://www.dropbox.com/s/qag9vzambhhkvxr/lip_jppnet_384.pb?dl=0
//

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


static Mat parse_human(const Mat &image, const std::string &model, int backend=dnn::DNN_BACKEND_DEFAULT, int target=dnn::DNN_TARGET_CPU) {
    // this network expects an image and a flipped copy as input
    Mat flipped;
    flip(image, flipped, 1);
    std::vector<Mat> batch;
    batch.push_back(image);
    batch.push_back(flipped);
    Mat blob = dnn::blobFromImages(batch, 1.0, Size(), Scalar(104.00698793, 116.66876762, 122.67891434));

    dnn::Net net = dnn::readNet(model);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.setInput(blob);
    Mat out = net.forward();
    // expected output: [2, 20, 384, 384], (2 lists(orig, flipped) of 20 body part heatmaps 384x384)

    // LIP classes:
    // 0 Background, 1 Hat, 2 Hair, 3 Glove, 4 Sunglasses, 5 UpperClothes, 6 Dress, 7 Coat, 8 Socks, 9 Pants
    // 10 Jumpsuits, 11 Scarf, 12 Skirt, 13 Face, 14 LeftArm, 15 RightArm, 16 LeftLeg, 17 RightLeg, 18 LeftShoe. 19 RightShoe
    Vec3b colors[] = {
        Vec3b(0, 0, 0), Vec3b(128, 0, 0), Vec3b(255, 0, 0), Vec3b(0, 85, 0), Vec3b(170, 0, 51), Vec3b(255, 85, 0),
        Vec3b(0, 0, 85), Vec3b(0, 119, 221), Vec3b(85, 85, 0), Vec3b(0, 85, 85), Vec3b(85, 51, 0), Vec3b(52, 86, 128),
        Vec3b(0, 128, 0), Vec3b(0, 0, 255), Vec3b(51, 170, 221), Vec3b(0, 255, 255), Vec3b(85, 255, 170),
        Vec3b(170, 255, 85), Vec3b(255, 255, 0), Vec3b(255, 170, 0)
    };

    Mat segm(image.size(), CV_8UC3, Scalar(0,0,0));
    Mat maxval(image.size(), CV_32F, Scalar(0));

    // iterate over body part heatmaps (LIP classes)
    for (int i=0; i<out.size[1]; i++) {
        // resize heatmaps to original image size
        // "head" is  the original image result, "tail" the flipped copy
        Mat head, h(out.size[2], out.size[3], CV_32F, out.ptr<float>(0,i));
        resize(h, head, image.size());

        // we have to swap the last 3 pairs in the "tail" list
        static int tail_order[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,14,17,16,19,18};
        Mat tail, t(out.size[2], out.size[3], CV_32F, out.ptr<float>(1,tail_order[i]));
        resize(t, tail, image.size());
        flip(tail, tail, 1);

        // mix original and flipped result
        Mat avg = (head + tail) * 0.5;

        // write color if prob value > maxval
        Mat cmask;
        compare(avg, maxval, cmask, CMP_GT);
        segm.setTo(colors[i], cmask);

        // keep largest values for next iteration
        max(avg, maxval, maxval);
    }
    cvtColor(segm, segm, COLOR_RGB2BGR);
    return segm;
}

int main(int argc, char**argv)
{
    CommandLineParser parser(argc,argv,
        "{help    h |                 | show help screen / args}"
        "{image   i |                 | person image to process }"
        "{model   m |lip_jppnet_384.pb| network model}"
        "{backend b | 0               | Choose one of computation backends: "
                                         "0: automatically (by default), "
                                         "1: Halide language (http://halide-lang.org/), "
                                         "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                                         "3: OpenCV implementation }"
        "{target  t | 0               | Choose one of target computation devices: "
                                         "0: CPU target (by default), "
                                         "1: OpenCL, "
                                         "2: OpenCL fp16 (half-float precision), "
                                         "3: VPU }"
    );
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string model = parser.get<std::string>("model");
    std::string image = parser.get<std::string>("image");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");

    Mat input = imread(image);
    Mat segm = parse_human(input, model, backend, target);

    imshow("human parsing", segm);
    waitKey();
    return 0;
}

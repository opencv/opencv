#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videostab/videostab.hpp"

using namespace std;
using namespace cv;
using namespace cv::videostab;

Ptr<Stabilizer> stabilizer;
double outputFps;
string outputPath;

void run();
void printHelp();

void run()
{
    VideoWriter writer;
    Mat stabilizedFrame;

    while (!(stabilizedFrame = stabilizer->nextFrame()).empty())
    {
        if (!outputPath.empty())
        {
            if (!writer.isOpened())
                writer.open(outputPath, CV_FOURCC('X','V','I','D'), outputFps, stabilizedFrame.size());
            writer << stabilizedFrame;
        }
        imshow("stabilizedFrame", stabilizedFrame);
        char key = static_cast<char>(waitKey(3));
        if (key == 27)
            break;
    }

    cout << "\nfinished\n";
}


void printHelp()
{
    cout << "OpenCV video stabilizer.\n"
            "Usage: videostab <file_path> [arguments]\n\n"
            "Arguments:\n"
            "  -m, --model=(transl|transl_and_scale|affine)\n"
            "      Set motion model. The default is affine.\n"
            "  --outlier-ratio=<float_number>\n"
            "      Outliers ratio in motion estimation. The default is 0.5.\n"
            "  --min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1,\n"
            "      but you may want to increase it.\n"
            "  -r, --radius=<int_number>\n"
            "      Set smoothing radius. The default is 15.\n"
            "  --stdev=<float_number>\n"
            "      Set smoothing weights standard deviation. The default is sqrt(radius).\n"
            "  --deblur=(yes|no)\n"
            "      Do deblurring.\n"
            "  --deblur-sens=<float_number>\n"
            "      Set deblurring sensitivity (from 0 to +inf). The default is 0.1.\n"
            "  -t, --trim-ratio=<float_number>\n"
            "      Set trimming ratio (from 0 to 0.5). The default is 0.\n"
            "  --est-trim=(yes|no)\n"
            "      Estimate trim ratio automatically. The default is yes (that leads to two passes,\n"
            "      you can turn it off if you want to use one pass only).\n"
            "  --incl-constr=(yes|no)\n"
            "      Ensure the inclusion constraint is always satisfied. The default is no.\n"
            "  --border-mode=(replicate|const)\n"
            "      Set border extrapolation mode. The default is replicate.\n"
            "  --mosaic=(yes|no)\n"
            "      Do consistent mosaicing. The default is no.\n"
            "  --mosaic-stdev=<float_number>\n"
            "      Consistent mosaicing stdev threshold. The default is 10.\n"
            "  --motion-inpaint=(yes|no)\n"
            "      Do motion inpainting (requires GPU support). The default is no.\n"
            "  --color-inpaint=(yes|no)\n"
            "      Do color inpainting. The defailt is no.\n"
            "  -o, --output=<file_path>\n"
            "      Set output file path explicitely. The default is stabilized.avi.\n"
            "  --fps=<int_number>\n"
            "      Set output video FPS explicitely. By default the source FPS is used.\n"
            "  -h, --help\n"
            "      Print help.\n"
            "\n";
}


int main(int argc, const char **argv)
{
    try
    {
        const char *keys =
                "{ 1 | | | | }"
                "{ m | model | | }"
                "{ | min-inlier-ratio | | }"
                "{ | outlier-ratio | | }"
                "{ r | radius | | }"
                "{ | stdev | | }"
                "{ | deblur | | }"
                "{ | deblur-sens | | }"
                "{ | est-trim | | }"
                "{ t | trim-ratio | | }"
                "{ | incl-constr | | }"
                "{ | border-mode | | }"
                "{ | mosaic | | }"
                "{ | mosaic-stdev | | }"
                "{ | motion-inpaint | | }"
                "{ | color-inpaint | | }"
                "{ o | output | stabilized.avi | }"
                "{ | fps | | }"
                "{ h | help | false | }";
        CommandLineParser cmd(argc, argv, keys);

        // parse command arguments

        if (cmd.get<bool>("help"))
        {
            printHelp();
            return 0;
        }

        stabilizer = new Stabilizer();

        string inputPath = cmd.get<string>("1");
        if (inputPath.empty())
            throw runtime_error("specify video file path");

        VideoFileSource *frameSource = new VideoFileSource(inputPath);
        outputFps = frameSource->fps();
        stabilizer->setFrameSource(frameSource);
        cout << "frame count: " << frameSource->frameCount() << endl;

        PyrLkRobustMotionEstimator *motionEstimator = new PyrLkRobustMotionEstimator();
        if (cmd.get<string>("model") == "transl")           
            motionEstimator->setMotionModel(TRANSLATION);
        else if (cmd.get<string>("model") == "transl_and_scale")
            motionEstimator->setMotionModel(TRANSLATION_AND_SCALE);
        else if (cmd.get<string>("model") == "affine")
            motionEstimator->setMotionModel(AFFINE);
        else if (!cmd.get<string>("model").empty())
            throw runtime_error("unknow motion mode: " + cmd.get<string>("model"));        

        if (!cmd.get<string>("outlier-ratio").empty())
        {
            RansacParams ransacParams = motionEstimator->ransacParams();
            ransacParams.eps = cmd.get<float>("outlier-ratio");
            motionEstimator->setRansacParams(ransacParams);
        }

        if (!cmd.get<string>("min-inlier-ratio").empty())
            motionEstimator->setMinInlierRatio(cmd.get<float>("min-inlier-ratio"));

        stabilizer->setMotionEstimator(motionEstimator);

        int smoothRadius = -1;
        float smoothStdev = -1;
        if (!cmd.get<string>("radius").empty())
            smoothRadius = cmd.get<int>("radius");
        if (!cmd.get<string>("stdev").empty())
            smoothStdev = cmd.get<float>("stdev");
        if (smoothRadius > 0 && smoothStdev > 0)
            stabilizer->setMotionFilter(new GaussianMotionFilter(smoothRadius, smoothStdev));
        else if (smoothRadius > 0 && smoothStdev < 0)
            stabilizer->setMotionFilter(new GaussianMotionFilter(smoothRadius, sqrt(smoothRadius)));

        if (cmd.get<string>("deblur") == "yes")
        {
            WeightingDeblurer *deblurer = new WeightingDeblurer();
            if (!cmd.get<string>("deblur-sens").empty())
                deblurer->setSensitivity(cmd.get<float>("deblur-sens"));
            stabilizer->setDeblurer(deblurer);
        }

        if (!cmd.get<string>("est-trim").empty())
            stabilizer->setEstimateTrimRatio(cmd.get<string>("est-trim") == "yes");

        if (!cmd.get<string>("trim-ratio").empty())
            stabilizer->setTrimRatio(cmd.get<float>("trim-ratio"));

        if (!cmd.get<string>("incl-constr").empty())
            stabilizer->setInclusionConstraint(cmd.get<string>("incl-constr") == "yes");

        if (cmd.get<string>("border-mode") == "replicate")
            stabilizer->setBorderMode(BORDER_REPLICATE);
        else if (cmd.get<string>("border-mode") == "const")
            stabilizer->setBorderMode(BORDER_CONSTANT);
        else if (!cmd.get<string>("border-mode").empty())
            throw runtime_error("unknown border extrapolation mode: " + cmd.get<string>("border-mode"));

        InpaintingPipeline *inpainters = new InpaintingPipeline();
        if (cmd.get<string>("mosaic") == "yes")
        {
            ConsistentMosaicInpainter *inpainter = new ConsistentMosaicInpainter();
            if (!cmd.get<string>("mosaic-stdev").empty())
                inpainter->setStdevThresh(cmd.get<float>("mosaic-stdev"));
            inpainters->pushBack(inpainter);
        }
        if (cmd.get<string>("motion-inpaint") == "yes")
            inpainters->pushBack(new MotionInpainter());
        if (cmd.get<string>("color-inpaint") == "yes")
            inpainters->pushBack(new ColorAverageInpainter());
        if (!inpainters->empty())
            stabilizer->setInpainter(inpainters);

        stabilizer->setLog(new LogToStdout());

        outputPath = cmd.get<string>("output");

        if (!cmd.get<string>("fps").empty())
            outputFps = cmd.get<double>("fps");

        // run video processing
        run();
    }
    catch (const exception &e)
    {
        cout << e.what() << endl;
        stabilizer.release();
        return -1;
    }
    stabilizer.release();
    return 0;
}

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videostab/videostab.hpp"

#define arg(name) cmd.get<string>(name)
#define argb(name) cmd.get<bool>(name)
#define argi(name) cmd.get<int>(name)
#define argf(name) cmd.get<float>(name)
#define argd(name) cmd.get<double>(name)

using namespace std;
using namespace cv;
using namespace cv::videostab;

Ptr<IFrameSource> stabilizedFrames;
string saveMotionsPath;
double outputFps;
string outputPath;
bool quietMode;

void run();
void saveMotionsIfNecessary();
void printHelp();

class GlobalMotionReader : public GlobalMotionEstimatorBase
{
public:
    GlobalMotionReader(string path)
    {
        ifstream f(path.c_str());
        if (!f.is_open())
            throw runtime_error("can't open motions file: " + path);
        int size; f >> size;
        motions_.resize(size);
        for (int i = 0; i < size; ++i)
        {
            Mat_<float> M(3, 3);
            for (int l = 0; l < 3; ++l)
                for (int s = 0; s < 3; ++s)
                    f >> M(l,s);
            motions_[i] = M;
        }
        pos_ = 0;
    }

    virtual Mat estimate(const Mat &/*frame0*/, const Mat &/*frame1*/)
    {
        if (pos_ >= motions_.size())
        {
            stringstream text;
            text << "can't load motion between frames " << pos_ << " and " << pos_+1;
            throw runtime_error(text.str());
        }
        return motions_[pos_++];
    }

private:
    vector<Mat> motions_;
    size_t pos_;
};


void run()
{
    VideoWriter writer;
    Mat stabilizedFrame;
    int nframes = 0;

    while (!(stabilizedFrame = stabilizedFrames->nextFrame()).empty())
    {
        nframes++;
        if (!saveMotionsPath.empty())
            saveMotionsIfNecessary();
        if (!outputPath.empty())
        {
            if (!writer.isOpened())
                writer.open(outputPath, CV_FOURCC('X','V','I','D'), outputFps, stabilizedFrame.size());
            writer << stabilizedFrame;
        }
        if (!quietMode)
        {
            imshow("stabilizedFrame", stabilizedFrame);
            char key = static_cast<char>(waitKey(3));
            if (key == 27)
                break;
        }
    }

    cout << endl
         << "processed frames: " << nframes << endl
         << "finished\n";
}


void saveMotionsIfNecessary()
{
    static bool areMotionsSaved = false;
    if (!areMotionsSaved)
    {
        IFrameSource *frameSource = static_cast<IFrameSource*>(stabilizedFrames);
        TwoPassStabilizer *twoPassStabilizer = dynamic_cast<TwoPassStabilizer*>(frameSource);
        if (twoPassStabilizer)
        {
            ofstream f(saveMotionsPath.c_str());
            const vector<Mat> &motions = twoPassStabilizer->motions();
            f << motions.size() << endl;
            for (size_t i = 0; i < motions.size(); ++i)
            {
                Mat_<float> M = motions[i];
                for (int l = 0, k = 0; l < 3; ++l)
                    for (int s = 0; s < 3; ++s, ++k)
                        f << M(l,s) << " ";
                f << endl;
            }
        }
        areMotionsSaved = true;
        cout << "motions are saved";
    }
}


void printHelp()
{
    cout << "OpenCV video stabilizer.\n"
            "Usage: videostab <file_path> [arguments]\n\n"
            "Arguments:\n"
            "  -m, --model=(transl|transl_and_scale|linear_sim|affine)\n"
            "      Set motion model. The default is affine.\n"
            "  --outlier-ratio=<float_number>\n"
            "      Outliers ratio in motion estimation. The default is 0.5.\n"
            "  --min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1,\n"
            "      but you may want to increase it.\n\n"
            "  --save-motions=(<file_path>|no)\n"
            "      Save estimated motions into file. The default is no.\n"
            "  --load-motions=(<file_path>|no)\n"
            "      Load motions from file. The default is no.\n\n"
            "  -r, --radius=<int_number>\n"
            "      Set sliding window radius. The default is 15.\n"
            "  --stdev=(<float_number>|auto)\n"
            "      Set smoothing weights standard deviation. The default is sqrt(radius),\n"
            "      i.e. auto.\n\n"
            "  --deblur=(yes|no)\n"
            "      Do deblurring.\n"
            "  --deblur-sens=<float_number>\n"
            "      Set deblurring sensitivity (from 0 to +inf). The default is 0.1.\n\n"
            "  -t, --trim-ratio=<float_number>\n"
            "      Set trimming ratio (from 0 to 0.5). The default is 0.1.\n"
            "  --est-trim=(yes|no)\n"
            "      Estimate trim ratio automatically. The default is yes (that leads to two passes,\n"
            "      you can turn it off if you want to use one pass only).\n"
            "  --incl-constr=(yes|no)\n"
            "      Ensure the inclusion constraint is always satisfied. The default is no.\n\n"
            "  --border-mode=(replicate|reflect|const)\n"
            "      Set border extrapolation mode. The default is replicate.\n\n"
            "  --mosaic=(yes|no)\n"
            "      Do consistent mosaicing. The default is no.\n"
            "  --mosaic-stdev=<float_number>\n"
            "      Consistent mosaicing stdev threshold. The default is 10.0.\n\n"
            "  --motion-inpaint=(yes|no)\n"
            "      Do motion inpainting (requires GPU support). The default is no.\n"
            "  --dist-thresh=<float_number>\n"
            "      Estimated flow distance threshold for motion inpainting. The default is 5.0.\n\n"
            "  --color-inpaint=(no|average|ns|telea)\n"
            "      Do color inpainting. The defailt is no.\n"
            "  --color-inpaint-radius=<float_number>\n"
            "      Set color inpainting radius (for ns and telea options only).\n"
            "      The default is 2.0\n\n"
            "  -o, --output=(no|<file_path>)\n"
            "      Set output file path explicitely. The default is stabilized.avi.\n"
            "  --fps=(<int_number>|auto)\n"
            "      Set output video FPS explicitely. By default the source FPS is used.\n"
            "  -q, --quiet\n"
            "      Don't show output video frames.\n\n"
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
                "{ m | model | affine| }"
                "{ | min-inlier-ratio | 0.1 | }"
                "{ | outlier-ratio | 0.5 | }"
                "{ | save-motions | no | }"
                "{ | load-motions | no | }"
                "{ r | radius | 15 | }"
                "{ | stdev | auto | }"
                "{ | deblur | no | }"
                "{ | deblur-sens | 0.1 | }"
                "{ | est-trim | yes | }"
                "{ t | trim-ratio | 0.1 | }"
                "{ | incl-constr | no | }"
                "{ | border-mode | replicate | }"
                "{ | mosaic | no | }"
                "{ | mosaic-stdev | 10.0 | }"
                "{ | motion-inpaint | no | }"
                "{ | dist-thresh | 5.0 | }"
                "{ | color-inpaint | no | }"
                "{ | color-inpaint-radius | 2 | }"
                "{ o | output | stabilized.avi | }"
                "{ | fps | auto | }"
                "{ q | quiet | false | }"
                "{ h | help | false | }";
        CommandLineParser cmd(argc, argv, keys);

        // parse command arguments

        if (argb("help"))
        {
            printHelp();
            return 0;
        }               

        StabilizerBase *stabilizer;

        bool isTwoPass = arg("est-trim") == "yes" || arg("save-motions") != "no";
        if (isTwoPass)
        {
            TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
            stabilizer = twoPassStabilizer;
            twoPassStabilizer->setEstimateTrimRatio(arg("est-trim") == "yes");
            if (arg("stdev") == "auto")
                twoPassStabilizer->setMotionStabilizer(new GaussianMotionFilter(argi("radius")));
            else
                twoPassStabilizer->setMotionStabilizer(new GaussianMotionFilter(argi("radius"), argf("stdev")));
        }
        else
        {
            OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
            stabilizer = onePassStabilizer;
            if (arg("stdev") == "auto")
                onePassStabilizer->setMotionFilter(new GaussianMotionFilter(argi("radius")));
            else
                onePassStabilizer->setMotionFilter(new GaussianMotionFilter(argi("radius"), argf("stdev")));
        }
        stabilizedFrames = dynamic_cast<IFrameSource*>(stabilizer);

        string inputPath = arg("1");
        if (inputPath.empty()) throw runtime_error("specify video file path");

        VideoFileSource *source = new VideoFileSource(inputPath);
        cout << "frame count (rough): " << source->count() << endl;
        if (arg("fps") == "auto") outputFps = source->fps();  else outputFps = argd("fps");
        stabilizer->setFrameSource(source);

        if (arg("load-motions") == "no")
        {
            RansacParams ransac;
            PyrLkRobustMotionEstimator *est = new PyrLkRobustMotionEstimator();
            Ptr<GlobalMotionEstimatorBase> est_(est);
            if (arg("model") == "transl")
            {
                est->setMotionModel(TRANSLATION);
                ransac = RansacParams::translation2dMotionStd();
            }
            else if (arg("model") == "transl_and_scale")
            {
                est->setMotionModel(TRANSLATION_AND_SCALE);
                ransac = RansacParams::translationAndScale2dMotionStd();
            }
            else if (arg("model") == "linear_sim")
            {
                est->setMotionModel(LINEAR_SIMILARITY);
                ransac = RansacParams::linearSimilarity2dMotionStd();
            }
            else if (arg("model") == "affine")
            {
                est->setMotionModel(AFFINE);
                ransac = RansacParams::affine2dMotionStd();
            }            
            else
                throw runtime_error("unknown motion model: " + arg("model"));
            ransac.eps = argf("outlier-ratio");
            est->setRansacParams(ransac);
            est->setMinInlierRatio(argf("min-inlier-ratio"));
            stabilizer->setMotionEstimator(est_);
        }
        else
            stabilizer->setMotionEstimator(new GlobalMotionReader(arg("load-motions")));

        if (arg("save-motions") != "no")
            saveMotionsPath = arg("save-motions");

        stabilizer->setRadius(argi("radius"));
        if (arg("deblur") == "yes")
        {
            WeightingDeblurer *deblurer = new WeightingDeblurer();
            deblurer->setRadius(argi("radius"));
            deblurer->setSensitivity(argf("deblur-sens"));
            stabilizer->setDeblurer(deblurer);
        }

        stabilizer->setTrimRatio(argf("trim-ratio"));
        stabilizer->setCorrectionForInclusion(arg("incl-constr") == "yes");

        if (arg("border-mode") == "reflect")
            stabilizer->setBorderMode(BORDER_REFLECT);
        else if (arg("border-mode") == "replicate")
            stabilizer->setBorderMode(BORDER_REPLICATE);
        else if (arg("border-mode") == "const")
            stabilizer->setBorderMode(BORDER_CONSTANT);
        else
            throw runtime_error("unknown border extrapolation mode: "
                                 + cmd.get<string>("border-mode"));

        InpaintingPipeline *inpainters = new InpaintingPipeline();
        Ptr<InpainterBase> inpainters_(inpainters);
        if (arg("mosaic") == "yes")
        {
            ConsistentMosaicInpainter *inp = new ConsistentMosaicInpainter();
            inp->setStdevThresh(argf("mosaic-stdev"));
            inpainters->pushBack(inp);
        }
        if (arg("motion-inpaint") == "yes")
        {
            MotionInpainter *inp = new MotionInpainter();
            inp->setDistThreshold(argf("dist-thresh"));
            inpainters->pushBack(inp);
        }
        if (arg("color-inpaint") == "average")
            inpainters->pushBack(new ColorAverageInpainter());
        else if (arg("color-inpaint") == "ns")
            inpainters->pushBack(new ColorInpainter(INPAINT_NS, argd("color-inpaint-radius")));
        else if (arg("color-inpaint") == "telea")
            inpainters->pushBack(new ColorInpainter(INPAINT_TELEA, argd("color-inpaint-radius")));
        else if (arg("color-inpaint") != "no")
            throw runtime_error("unknown color inpainting method: " + arg("color-inpaint"));
        if (!inpainters->empty())
        {
            inpainters->setRadius(argi("radius"));
            stabilizer->setInpainter(inpainters_);
        }

        stabilizer->setLog(new LogToStdout());

        if (arg("output") != "no")
            outputPath = arg("output");

        quietMode = argb("quite");

        run();
    }
    catch (const exception &e)
    {
        cout << "error: " << e.what() << endl;
        stabilizedFrames.release();
        return -1;
    }
    stabilizedFrames.release();
    return 0;
}

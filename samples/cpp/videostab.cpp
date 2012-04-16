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


void run()
{
    VideoWriter writer;
    Mat stabilizedFrame;
    int nframes = 0;

    while (!(stabilizedFrame = stabilizedFrames->nextFrame()).empty())
    {
        nframes++;
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


void printHelp()
{
    cout << "OpenCV video stabilizer.\n"
            "Usage: videostab <file_path> [arguments]\n\n"
            "Arguments:\n"
            "  -m, --model=(transl|transl_and_scale|linear_sim|affine|homography)\n"
            "      Set motion model. The default is affine.\n"
            "  --subset=(<int_number>|auto)\n"
            "      Number of random samples per one motion hypothesis. The default is auto.\n"
            "  --outlier-ratio=<float_number>\n"
            "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
            "  --min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
            "  --nkps=<int_number>\n"
            "      Number of keypoints to find in each frame. The default is 1000.\n"
            "  --extra-kps=<int_number>\n"
            "      Extra keypoint grid size for motion estimation. The default is 0.\n\n"
            "  -sm, --save-motions=(<file_path>|no)\n"
            "      Save estimated motions into file. The default is no.\n"
            "  -lm, --load-motions=(<file_path>|no)\n"
            "      Load motions from file. The default is no.\n\n"
            "  -r, --radius=<int_number>\n"
            "      Set sliding window radius. The default is 15.\n"
            "  --stdev=(<float_number>|auto)\n"
            "      Set smoothing weights standard deviation. The default is auto\n"
            "      (i.e. sqrt(radius)).\n"
            "  -lp, --lp-stab=(yes|no)\n"
            "      Turn on/off linear programming based stabilization method.\n"
            "  --lp-trim-ratio=(<float_number>|auto)\n"
            "      Trimming ratio used in linear programming based method.\n"
            "  --lp-w1=(<float_number>|1)\n"
            "      1st derivative weight. The default is 1.\n"
            "  --lp-w2=(<float_number>|10)\n"
            "      2nd derivative weight. The default is 10.\n"
            "  --lp-w3=(<float_number>|100)\n"
            "      3rd derivative weight. The default is 100.\n"
            "  --lp-w4=(<float_number>|100)\n"
            "      Non-translation motion components weight. The default is 100.\n\n"
            "  --deblur=(yes|no)\n"
            "      Do deblurring.\n"
            "  --deblur-sens=<float_number>\n"
            "      Set deblurring sensitivity (from 0 to +inf). The default is 0.1.\n\n"
            "  -t, --trim-ratio=<float_number>\n"
            "      Set trimming ratio (from 0 to 0.5). The default is 0.1.\n"
            "  -et, --est-trim=(yes|no)\n"
            "      Estimate trim ratio automatically. The default is yes.\n"
            "  -ic, --incl-constr=(yes|no)\n"
            "      Ensure the inclusion constraint is always satisfied. The default is no.\n\n"
            "  -bm, --border-mode=(replicate|reflect|const)\n"
            "      Set border extrapolation mode. The default is replicate.\n\n"
            "  --mosaic=(yes|no)\n"
            "      Do consistent mosaicing. The default is no.\n"
            "  --mosaic-stdev=<float_number>\n"
            "      Consistent mosaicing stdev threshold. The default is 10.0.\n\n"
            "  -mi, --motion-inpaint=(yes|no)\n"
            "      Do motion inpainting (requires GPU support). The default is no.\n"
            "  --mi-dist-thresh=<float_number>\n"
            "      Estimated flow distance threshold for motion inpainting. The default is 5.0.\n\n"
            "  -ci, --color-inpaint=(no|average|ns|telea)\n"
            "      Do color inpainting. The defailt is no.\n"
            "  --ci-radius=<float_number>\n"
            "      Set color inpainting radius (for ns and telea options only).\n"
            "      The default is 2.0\n\n"
            "  -ws, --wobble-suppress=(yes|no)\n"
            "      Perform wobble suppression. The default is no.\n"
            "  --ws-period=<int_number>\n"
            "      Set wobble suppression period. The default is 30.\n"
            "  --ws-model=(transl|transl_and_scale|linear_sim|affine|homography)\n"
            "      Set wobble suppression motion model (must have more DOF than motion \n"
            "      estimation model). The default is homography.\n"
            "  --ws-subset=(<int_number>|auto)\n"
            "      Number of random samples per one motion hypothesis. The default is auto.\n"
            "  --ws-outlier-ratio=<float_number>\n"
            "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
            "  --ws-min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
            "  --ws-nkps=<int_number>\n"
            "      Number of keypoints to find in each frame. The default is 1000.\n"
            "  --ws-extra-kps=<int_number>\n"
            "      Extra keypoint grid size for motion estimation. The default is 0.\n\n"
            "  -sm2, --save-motions2=(<file_path>|no)\n"
            "      Save motions estimated for wobble suppression. The default is no.\n"
            "  -lm2, --load-motions2=(<file_path>|no)\n"
            "      Load motions for wobble suppression from file. The default is no.\n\n"
            "  -o, --output=(no|<file_path>)\n"
            "      Set output file path explicitely. The default is stabilized.avi.\n"
            "  --fps=(<float_number>|auto)\n"
            "      Set output video FPS explicitely. By default the source FPS is used (auto).\n"
            "  -q, --quiet\n"
            "      Don't show output video frames.\n\n"
            "  -h, --help\n"
            "      Print help.\n\n"
            "Note: some argument configurations lead to two passes, some to single pass.\n\n";
}


int main(int argc, const char **argv)
{
    try
    {
        const char *keys =
                "{ 1 | | | | }"
                "{ m | model | affine| }"
                "{ | subset | auto | }"
                "{ | outlier-ratio | 0.5 | }"
                "{ | min-inlier-ratio | 0.1 | }"
                "{ | nkps | 1000 | }"
                "{ | extra-kps | 0 | }"
                "{ sm | save-motions | no | }"
                "{ lm | load-motions | no | }"
                "{ r | radius | 15 | }"
                "{ | stdev | auto | }"
                "{ lp | lp-stab | no | }"
                "{ | lp-trim-ratio | auto | }"
                "{ | lp-w1 | 1 | }"
                "{ | lp-w2 | 10 | }"
                "{ | lp-w3 | 100 | }"
                "{ | lp-w4 | 100 | }"
                "{ | deblur | no | }"
                "{ | deblur-sens | 0.1 | }"
                "{ et | est-trim | yes | }"
                "{ t | trim-ratio | 0.1 | }"
                "{ ic | incl-constr | no | }"
                "{ bm | border-mode | replicate | }"
                "{ | mosaic | no | }"
                "{ ms | mosaic-stdev | 10.0 | }"
                "{ mi | motion-inpaint | no | }"
                "{ | mi-dist-thresh | 5.0 | }"
                "{ ci | color-inpaint | no | }"
                "{ | ci-radius | 2 | }"
                "{ ws | wobble-suppress | no | }"
                "{ | ws-period | 30 | }"
                "{ | ws-model | homography | }"
                "{ | ws-subset | auto | }"
                "{ | ws-outlier-ratio | 0.5 | }"
                "{ | ws-min-inlier-ratio | 0.1 | }"
                "{ | ws-nkps | 1000 | }"
                "{ | ws-extra-kps | 0 | }"
                "{ sm2 | save-motions2 | no | }"
                "{ lm2 | load-motions2 | no | }"
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

        string inputPath = arg("1");
        if (inputPath.empty()) throw runtime_error("specify video file path");

        VideoFileSource *source = new VideoFileSource(inputPath);
        cout << "frame count (rough): " << source->count() << endl;
        if (arg("fps") == "auto") outputFps = source->fps();  else outputFps = argd("fps");

        StabilizerBase *stabilizer;

        bool isTwoPass =
                arg("est-trim") == "yes" || arg("wobble-suppress") == "yes" || arg("lp-stab") == "yes";

        if (isTwoPass)
        {
            TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
            stabilizer = twoPassStabilizer;
            twoPassStabilizer->setEstimateTrimRatio(arg("est-trim") == "yes");
            if (arg("lp-stab") == "yes")
            {
                LpMotionStabilizer *stab = new LpMotionStabilizer();
                stab->setFrameSize(Size(source->width(), source->height()));
                stab->setTrimRatio(arg("lp-trim-ratio") == "auto" ? argf("trim-ratio") : argf("lp-trim-ratio"));
                stab->setWeight1(argf("lp-w1"));
                stab->setWeight2(argf("lp-w2"));
                stab->setWeight3(argf("lp-w3"));
                stab->setWeight4(argf("lp-w4"));
                twoPassStabilizer->setMotionStabilizer(stab);
            }
            else if (arg("stdev") == "auto")
                twoPassStabilizer->setMotionStabilizer(new GaussianMotionFilter(argi("radius")));
            else
                twoPassStabilizer->setMotionStabilizer(new GaussianMotionFilter(argi("radius"), argf("stdev")));
            if (arg("wobble-suppress") == "yes")
            {
                MoreAccurateMotionWobbleSuppressor *ws = new MoreAccurateMotionWobbleSuppressor();
                twoPassStabilizer->setWobbleSuppressor(ws);
                ws->setPeriod(argi("ws-period"));

                PyrLkRobustMotionEstimator *est = 0;

                if (arg("ws-model") == "transl")
                    est = new PyrLkRobustMotionEstimator(MM_TRANSLATION);
                else if (arg("ws-model") == "transl_and_scale")
                    est = new PyrLkRobustMotionEstimator(MM_TRANSLATION_AND_SCALE);
                else if (arg("ws-model") == "linear_sim")
                    est = new PyrLkRobustMotionEstimator(MM_LINEAR_SIMILARITY);
                else if (arg("ws-model") == "affine")
                    est = new PyrLkRobustMotionEstimator(MM_AFFINE);
                else if (arg("ws-model") == "homography")
                    est = new PyrLkRobustMotionEstimator(MM_HOMOGRAPHY);
                else
                {
                    delete est;
                    throw runtime_error("unknown wobble suppression motion model: " + arg("ws-model"));
                }

                est->setDetector(new GoodFeaturesToTrackDetector(argi("ws-nkps")));
                RansacParams ransac = est->ransacParams();
                if (arg("ws-subset") != "auto")
                    ransac.size = argi("ws-subset");
                ransac.eps = argf("ws-outlier-ratio");
                est->setRansacParams(ransac);
                est->setMinInlierRatio(argf("ws-min-inlier-ratio"));
                est->setGridSize(Size(argi("ws-extra-kps"), argi("ws-extra-kps")));
                ws->setMotionEstimator(est);                

                MotionModel model = est->motionModel();
                if (arg("load-motions2") != "no")
                {
                    ws->setMotionEstimator(new FromFileMotionReader(arg("load-motions2")));
                    ws->motionEstimator()->setMotionModel(model);
                }
                if (arg("save-motions2") != "no")
                {
                    ws->setMotionEstimator(new ToFileMotionWriter(arg("save-motions2"), ws->motionEstimator()));
                    ws->motionEstimator()->setMotionModel(model);
                }
             }
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

        stabilizer->setFrameSource(source);
        stabilizedFrames = dynamic_cast<IFrameSource*>(stabilizer);

        PyrLkRobustMotionEstimator *est = 0;

        if (arg("model") == "transl")
            est = new PyrLkRobustMotionEstimator(MM_TRANSLATION);
        else if (arg("model") == "transl_and_scale")
            est = new PyrLkRobustMotionEstimator(MM_TRANSLATION_AND_SCALE);
        else if (arg("model") == "linear_sim")
            est = new PyrLkRobustMotionEstimator(MM_LINEAR_SIMILARITY);
        else if (arg("model") == "affine")
            est = new PyrLkRobustMotionEstimator(MM_AFFINE);
        else if (arg("model") == "homography")
            est = new PyrLkRobustMotionEstimator(MM_HOMOGRAPHY);
        else
        {
            delete est;
            throw runtime_error("unknown motion model: " + arg("model"));
        }

        est->setDetector(new GoodFeaturesToTrackDetector(argi("nkps")));
        RansacParams ransac = est->ransacParams();
        if (arg("subset") != "auto")
            ransac.size = argi("subset");
        ransac.eps = argf("outlier-ratio");
        est->setRansacParams(ransac);
        est->setMinInlierRatio(argf("min-inlier-ratio"));
        est->setGridSize(Size(argi("extra-kps"), argi("extra-kps")));
        stabilizer->setMotionEstimator(est);

        MotionModel model = stabilizer->motionEstimator()->motionModel();
        if (arg("load-motions") != "no")
        {
            stabilizer->setMotionEstimator(new FromFileMotionReader(arg("load-motions")));
            stabilizer->motionEstimator()->setMotionModel(model);
        }
        if (arg("save-motions") != "no")
        {
            stabilizer->setMotionEstimator(new ToFileMotionWriter(arg("save-motions"), stabilizer->motionEstimator()));
            stabilizer->motionEstimator()->setMotionModel(model);
        }

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
            inp->setDistThreshold(argf("mi-dist-thresh"));
            inpainters->pushBack(inp);
        }
        if (arg("color-inpaint") == "average")
            inpainters->pushBack(new ColorAverageInpainter());
        else if (arg("color-inpaint") == "ns")
            inpainters->pushBack(new ColorInpainter(INPAINT_NS, argd("ci-radius")));
        else if (arg("color-inpaint") == "telea")
            inpainters->pushBack(new ColorInpainter(INPAINT_TELEA, argd("ci-radius")));
        else if (arg("color-inpaint") != "no")
            throw runtime_error("unknown color inpainting method: " + arg("color-inpaint"));
        if (!inpainters->empty())
        {
            inpainters->setRadius(argi("radius"));
            stabilizer->setInpainter(inpainters_);
        }       

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

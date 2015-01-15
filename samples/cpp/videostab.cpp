#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videostab.hpp"
#include "opencv2/opencv_modules.hpp"

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
MotionModel motionModel(const string &str);


void run()
{
    VideoWriter writer;
    Mat stabilizedFrame;
    int nframes = 0;

    // for each stabilized frame
    while (!(stabilizedFrame = stabilizedFrames->nextFrame()).empty())
    {
        nframes++;

        // init writer (once) and save stabilized frame
        if (!outputPath.empty())
        {
            if (!writer.isOpened())
                writer.open(outputPath, VideoWriter::fourcc('X','V','I','D'),
                            outputFps, stabilizedFrame.size());
            writer << stabilizedFrame;
        }

        // show stabilized frame
        if (!quietMode)
        {
            imshow("stabilizedFrame", stabilizedFrame);
            char key = static_cast<char>(waitKey(3));
            if (key == 27) { cout << endl; break; }
        }
    }

    cout << "processed frames: " << nframes << endl
         << "finished\n";
}


void printHelp()
{
    cout << "OpenCV video stabilizer.\n"
            "Usage: videostab <file_path> [arguments]\n\n"
            "Arguments:\n"
            "  -m, --model=(transl|transl_and_scale|rigid|similarity|affine|homography)\n"
            "      Set motion model. The default is affine.\n"
            "  -lp, --lin-prog-motion-est=(yes|no)\n"
            "      Turn on/off LP based motion estimation. The default is no.\n"
            "  --subset=(<int_number>|auto)\n"
            "      Number of random samples per one motion hypothesis. The default is auto.\n"
            "  --thresh=(<float_number>|auto)\n"
            "      Maximum error to classify match as inlier. The default is auto.\n"
            "  --outlier-ratio=<float_number>\n"
            "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
            "  --min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
            "  --nkps=<int_number>\n"
            "      Number of keypoints to find in each frame. The default is 1000.\n"
            "  --local-outlier-rejection=(yes|no)\n"
            "      Perform local outlier rejection. The default is no.\n\n"
            "  -sm, --save-motions=(<file_path>|no)\n"
            "      Save estimated motions into file. The default is no.\n"
            "  -lm, --load-motions=(<file_path>|no)\n"
            "      Load motions from file. The default is no.\n\n"
            "  -r, --radius=<int_number>\n"
            "      Set sliding window radius. The default is 15.\n"
            "  --stdev=(<float_number>|auto)\n"
            "      Set smoothing weights standard deviation. The default is auto\n"
            "      (i.e. sqrt(radius)).\n"
            "  -lps, --lin-prog-stab=(yes|no)\n"
            "      Turn on/off linear programming based stabilization method.\n"
            "  --lps-trim-ratio=(<float_number>|auto)\n"
            "      Trimming ratio used in linear programming based method.\n"
            "  --lps-w1=(<float_number>|1)\n"
            "      1st derivative weight. The default is 1.\n"
            "  --lps-w2=(<float_number>|10)\n"
            "      2nd derivative weight. The default is 10.\n"
            "  --lps-w3=(<float_number>|100)\n"
            "      3rd derivative weight. The default is 100.\n"
            "  --lps-w4=(<float_number>|100)\n"
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
            "      Do motion inpainting (requires CUDA support). The default is no.\n"
            "  --mi-dist-thresh=<float_number>\n"
            "      Estimated flow distance threshold for motion inpainting. The default is 5.0.\n\n"
            "  -ci, --color-inpaint=(no|average|ns|telea)\n"
            "      Do color inpainting. The defailt is no.\n"
            "  --ci-radius=<float_number>\n"
            "      Set color inpainting radius (for ns and telea options only).\n"
            "      The default is 2.0\n\n"
            "  -ws, --wobble-suppress=(yes|no)\n"
            "      Perform wobble suppression. The default is no.\n"
            "  --ws-lp=(yes|no)\n"
            "      Turn on/off LP based motion estimation. The default is no.\n"
            "  --ws-period=<int_number>\n"
            "      Set wobble suppression period. The default is 30.\n"
            "  --ws-model=(transl|transl_and_scale|rigid|similarity|affine|homography)\n"
            "      Set wobble suppression motion model (must have more DOF than motion \n"
            "      estimation model). The default is homography.\n"
            "  --ws-subset=(<int_number>|auto)\n"
            "      Number of random samples per one motion hypothesis. The default is auto.\n"
            "  --ws-thresh=(<float_number>|auto)\n"
            "      Maximum error to classify match as inlier. The default is auto.\n"
            "  --ws-outlier-ratio=<float_number>\n"
            "      Motion estimation outlier ratio hypothesis. The default is 0.5.\n"
            "  --ws-min-inlier-ratio=<float_number>\n"
            "      Minimum inlier ratio to decide if estimated motion is OK. The default is 0.1.\n"
            "  --ws-nkps=<int_number>\n"
            "      Number of keypoints to find in each frame. The default is 1000.\n"
            "  --ws-local-outlier-rejection=(yes|no)\n"
            "      Perform local outlier rejection. The default is no.\n\n"
            "  -sm2, --save-motions2=(<file_path>|no)\n"
            "      Save motions estimated for wobble suppression. The default is no.\n"
            "  -lm2, --load-motions2=(<file_path>|no)\n"
            "      Load motions for wobble suppression from file. The default is no.\n\n"
            "  -gpu=(yes|no)\n"
            "      Use CUDA optimization whenever possible. The default is no.\n\n"
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

// motion estimator builders are for concise creation of motion estimators

class IMotionEstimatorBuilder
{
public:
    virtual ~IMotionEstimatorBuilder() {}
    virtual Ptr<ImageMotionEstimatorBase> build() = 0;
protected:
    IMotionEstimatorBuilder(CommandLineParser &command) : cmd(command) {}
    CommandLineParser cmd;
};


class MotionEstimatorRansacL2Builder : public IMotionEstimatorBuilder
{
public:
    MotionEstimatorRansacL2Builder(CommandLineParser &command, bool use_gpu, const string &_prefix = "")
        : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

    virtual Ptr<ImageMotionEstimatorBase> build()
    {
        Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(motionModel(arg(prefix + "model")));

        RansacParams ransac = est->ransacParams();
        if (arg(prefix + "subset") != "auto")
            ransac.size = argi(prefix + "subset");
        if (arg(prefix + "thresh") != "auto")
            ransac.thresh = argf(prefix + "thresh");
        ransac.eps = argf(prefix + "outlier-ratio");
        est->setRansacParams(ransac);

        est->setMinInlierRatio(argf(prefix + "min-inlier-ratio"));

        Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
        if (arg(prefix + "local-outlier-rejection") == "yes")
        {
            Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
            RansacParams ransacParams = tblor->ransacParams();
            if (arg(prefix + "thresh") != "auto")
                ransacParams.thresh = argf(prefix + "thresh");
            tblor->setRansacParams(ransacParams);
            outlierRejector = tblor;
        }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
        if (gpu)
        {
            Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
            kbest->setOutlierRejector(outlierRejector);
            return kbest;
        }
#endif

        Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
        kbest->setDetector(GFTTDetector::create(argi(prefix + "nkps")));
        kbest->setOutlierRejector(outlierRejector);
        return kbest;
    }
private:
    bool gpu;
    string prefix;
};


class MotionEstimatorL1Builder : public IMotionEstimatorBuilder
{
public:
    MotionEstimatorL1Builder(CommandLineParser &command, bool use_gpu, const string &_prefix = "")
        : IMotionEstimatorBuilder(command), gpu(use_gpu), prefix(_prefix) {}

    virtual Ptr<ImageMotionEstimatorBase> build()
    {
        Ptr<MotionEstimatorL1> est = makePtr<MotionEstimatorL1>(motionModel(arg(prefix + "model")));

        Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
        if (arg(prefix + "local-outlier-rejection") == "yes")
        {
            Ptr<TranslationBasedLocalOutlierRejector> tblor = makePtr<TranslationBasedLocalOutlierRejector>();
            RansacParams ransacParams = tblor->ransacParams();
            if (arg(prefix + "thresh") != "auto")
                ransacParams.thresh = argf(prefix + "thresh");
            tblor->setRansacParams(ransacParams);
            outlierRejector = tblor;
        }

#if defined(HAVE_OPENCV_CUDAIMGPROC) && defined(HAVE_OPENCV_CUDAOPTFLOW)
        if (gpu)
        {
            Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
            kbest->setOutlierRejector(outlierRejector);
            return kbest;
        }
#endif

        Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
        kbest->setDetector(GFTTDetector::create(argi(prefix + "nkps")));
        kbest->setOutlierRejector(outlierRejector);
        return kbest;
    }
private:
    bool gpu;
    string prefix;
};


int main(int argc, const char **argv)
{
    try
    {
        const char *keys =
                "{ @1                       |           | }"
                "{ m  model                 | affine    | }"
                "{ lp lin-prog-motion-est   | no        | }"
                "{  subset                  | auto      | }"
                "{  thresh                  | auto | }"
                "{  outlier-ratio           | 0.5 | }"
                "{  min-inlier-ratio        | 0.1 | }"
                "{  nkps                    | 1000 | }"
                "{  extra-kps               | 0 | }"
                "{  local-outlier-rejection | no | }"
                "{ sm  save-motions         | no | }"
                "{ lm  load-motions         | no | }"
                "{ r  radius                | 15 | }"
                "{  stdev                   | auto | }"
                "{ lps  lin-prog-stab       | no | }"
                "{  lps-trim-ratio          | auto | }"
                "{  lps-w1                  | 1 | }"
                "{  lps-w2                  | 10 | }"
                "{  lps-w3                  | 100 | }"
                "{  lps-w4                  | 100 | }"
                "{  deblur                  | no | }"
                "{  deblur-sens             | 0.1 | }"
                "{ et  est-trim             | yes | }"
                "{ t  trim-ratio            | 0.1 | }"
                "{ ic  incl-constr          | no | }"
                "{ bm  border-mode          | replicate | }"
                "{  mosaic                  | no | }"
                "{ ms  mosaic-stdev         | 10.0 | }"
                "{ mi  motion-inpaint       | no | }"
                "{  mi-dist-thresh          | 5.0 | }"
                "{ ci color-inpaint         | no | }"
                "{  ci-radius               | 2 | }"
                "{ ws  wobble-suppress      | no | }"
                "{  ws-period               | 30 | }"
                "{  ws-model                | homography | }"
                "{  ws-subset               | auto | }"
                "{  ws-thresh               | auto | }"
                "{  ws-outlier-ratio        | 0.5 | }"
                "{  ws-min-inlier-ratio     | 0.1 | }"
                "{  ws-nkps                 | 1000 | }"
                "{  ws-extra-kps            | 0 | }"
                "{  ws-local-outlier-rejection | no | }"
                "{  ws-lp                   | no | }"
                "{ sm2 save-motions2        | no | }"
                "{ lm2 load-motions2        | no | }"
                "{ gpu                      | no | }"
                "{ o  output                | stabilized.avi | }"
                "{ fps                      | auto | }"
                "{ q quiet                  |  | }"
                "{ h help                   |  | }";
        CommandLineParser cmd(argc, argv, keys);

        // parse command arguments

        if (argb("help"))
        {
            printHelp();
            return 0;
        }

        if (arg("gpu") == "yes")
        {
            cout << "initializing GPU..."; cout.flush();
            Mat hostTmp = Mat::zeros(1, 1, CV_32F);
            cuda::GpuMat deviceTmp;
            deviceTmp.upload(hostTmp);
            cout << endl;
        }

        StabilizerBase *stabilizer = 0;

        // check if source video is specified

        string inputPath = arg(0);
        if (inputPath.empty())
            throw runtime_error("specify video file path");

        // get source video parameters

        Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
        cout << "frame count (rough): " << source->count() << endl;
        if (arg("fps") == "auto")
            outputFps = source->fps();
        else
            outputFps = argd("fps");

        // prepare motion estimation builders

        Ptr<IMotionEstimatorBuilder> motionEstBuilder;
        if (arg("lin-prog-motion-est") == "yes")
            motionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, arg("gpu") == "yes"));
        else
            motionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, arg("gpu") == "yes"));

        Ptr<IMotionEstimatorBuilder> wsMotionEstBuilder;
        if (arg("ws-lp") == "yes")
            wsMotionEstBuilder.reset(new MotionEstimatorL1Builder(cmd, arg("gpu") == "yes", "ws-"));
        else
            wsMotionEstBuilder.reset(new MotionEstimatorRansacL2Builder(cmd, arg("gpu") == "yes", "ws-"));

        // determine whether we must use one pass or two pass stabilizer
        bool isTwoPass =
                arg("est-trim") == "yes" || arg("wobble-suppress") == "yes" || arg("lin-prog-stab") == "yes";

        if (isTwoPass)
        {
            // we must use two pass stabilizer

            TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
            stabilizer = twoPassStabilizer;
            twoPassStabilizer->setEstimateTrimRatio(arg("est-trim") == "yes");

            // determine stabilization technique

            if (arg("lin-prog-stab") == "yes")
            {
                Ptr<LpMotionStabilizer> stab = makePtr<LpMotionStabilizer>();
                stab->setFrameSize(Size(source->width(), source->height()));
                stab->setTrimRatio(arg("lps-trim-ratio") == "auto" ? argf("trim-ratio") : argf("lps-trim-ratio"));
                stab->setWeight1(argf("lps-w1"));
                stab->setWeight2(argf("lps-w2"));
                stab->setWeight3(argf("lps-w3"));
                stab->setWeight4(argf("lps-w4"));
                twoPassStabilizer->setMotionStabilizer(stab);
            }
            else if (arg("stdev") == "auto")
                twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(argi("radius")));
            else
                twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(argi("radius"), argf("stdev")));

            // init wobble suppressor if necessary

            if (arg("wobble-suppress") == "yes")
            {
                Ptr<MoreAccurateMotionWobbleSuppressorBase> ws = makePtr<MoreAccurateMotionWobbleSuppressor>();
                if (arg("gpu") == "yes")
#ifdef HAVE_OPENCV_CUDAWARPING
                    ws = makePtr<MoreAccurateMotionWobbleSuppressorGpu>();
#else
                    throw runtime_error("OpenCV is built without CUDA support");
#endif

                ws->setMotionEstimator(wsMotionEstBuilder->build());
                ws->setPeriod(argi("ws-period"));
                twoPassStabilizer->setWobbleSuppressor(ws);

                MotionModel model = ws->motionEstimator()->motionModel();
                if (arg("load-motions2") != "no")
                {
                    ws->setMotionEstimator(makePtr<FromFileMotionReader>(arg("load-motions2")));
                    ws->motionEstimator()->setMotionModel(model);
                }
                if (arg("save-motions2") != "no")
                {
                    ws->setMotionEstimator(makePtr<ToFileMotionWriter>(arg("save-motions2"), ws->motionEstimator()));
                    ws->motionEstimator()->setMotionModel(model);
                }
            }
        }
        else
        {
            // we must use one pass stabilizer

            OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
            stabilizer = onePassStabilizer;
            if (arg("stdev") == "auto")
                onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(argi("radius")));
            else
                onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(argi("radius"), argf("stdev")));
        }

        stabilizer->setFrameSource(source);
        stabilizer->setMotionEstimator(motionEstBuilder->build());

        // cast stabilizer to simple frame source interface to read stabilized frames
        stabilizedFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));

        MotionModel model = stabilizer->motionEstimator()->motionModel();
        if (arg("load-motions") != "no")
        {
            stabilizer->setMotionEstimator(makePtr<FromFileMotionReader>(arg("load-motions")));
            stabilizer->motionEstimator()->setMotionModel(model);
        }
        if (arg("save-motions") != "no")
        {
            stabilizer->setMotionEstimator(makePtr<ToFileMotionWriter>(arg("save-motions"), stabilizer->motionEstimator()));
            stabilizer->motionEstimator()->setMotionModel(model);
        }

        stabilizer->setRadius(argi("radius"));

        // init deblurer
        if (arg("deblur") == "yes")
        {
            Ptr<WeightingDeblurer> deblurer = makePtr<WeightingDeblurer>();
            deblurer->setRadius(argi("radius"));
            deblurer->setSensitivity(argf("deblur-sens"));
            stabilizer->setDeblurer(deblurer);
        }

        // set up trimming paramters
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

        // init inpainter
        InpaintingPipeline *inpainters = new InpaintingPipeline();
        Ptr<InpainterBase> inpainters_(inpainters);
        if (arg("mosaic") == "yes")
        {
            Ptr<ConsistentMosaicInpainter> inp = makePtr<ConsistentMosaicInpainter>();
            inp->setStdevThresh(argf("mosaic-stdev"));
            inpainters->pushBack(inp);
        }
        if (arg("motion-inpaint") == "yes")
        {
            Ptr<MotionInpainter> inp = makePtr<MotionInpainter>();
            inp->setDistThreshold(argf("mi-dist-thresh"));
            inpainters->pushBack(inp);
        }
        if (arg("color-inpaint") == "average")
            inpainters->pushBack(makePtr<ColorAverageInpainter>());
        else if (arg("color-inpaint") == "ns")
            inpainters->pushBack(makePtr<ColorInpainter>(int(INPAINT_NS), argd("ci-radius")));
        else if (arg("color-inpaint") == "telea")
            inpainters->pushBack(makePtr<ColorInpainter>(int(INPAINT_TELEA), argd("ci-radius")));
        else if (arg("color-inpaint") != "no")
            throw runtime_error("unknown color inpainting method: " + arg("color-inpaint"));
        if (!inpainters->empty())
        {
            inpainters->setRadius(argi("radius"));
            stabilizer->setInpainter(inpainters_);
        }

        if (arg("output") != "no")
            outputPath = arg("output");

        quietMode = argb("quiet");

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


MotionModel motionModel(const string &str)
{
    if (str == "transl")
        return MM_TRANSLATION;
    if (str == "transl_and_scale")
        return MM_TRANSLATION_AND_SCALE;
    if (str == "rigid")
        return MM_RIGID;
    if (str == "similarity")
        return MM_SIMILARITY;
    if (str == "affine")
        return MM_AFFINE;
    if (str == "homography")
        return MM_HOMOGRAPHY;
    throw runtime_error("unknown motion model: " + str);
}

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility_lib/utility_lib.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define PARAM_METHOD "--method"
#define PARAM_OFFSET "--offset"

//#define USE_OPENGL

#ifdef USE_OPENGL
    struct DrawData
    {
        GlTexture tex;
        GlArrays arr;
        string method;
        string total_fps;
        string proc_fps;
    };
#endif

class App : public BaseApp
{
public:
    enum Method
    {
        BROX,
        LK_DENSE,
        LK_SPARSE_GPU, LK_SPARSE_CPU,
        FARNEBACK_GPU, FARNEBACK_CPU
    } method;

    App() : method(BROX),
            the_same_video_offset(1)
    {}

protected:
    void process();
    bool parseCmdArgs(int& i, int argc, const char* argv[]);
    void printHelp();
    bool processKey(int key);

private:
    int the_same_video_offset;

#ifdef USE_OPENGL
    DrawData draw_data;
#endif
};

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == PARAM_METHOD)
    {
        ++i;

        if (i >= argc)
        {
            ostringstream msg;
            msg << "Missing value after " << PARAM_METHOD;
            throw runtime_error(msg.str());
        }

        string m = argv[i];

        if (m == "brox")
            method = BROX;
        else if (m == "lk_dense")
            method = LK_DENSE;
        else if (m == "lk_sparse_gpu")
            method = LK_SPARSE_GPU;
        else if (m == "lk_sparse_cpu")
            method = LK_SPARSE_CPU;
        else if (m == "farneback_cpu")
            method = FARNEBACK_CPU;
        else if (m == "farneback_gpu")
            method = FARNEBACK_GPU;
        else
        {
            ostringstream msg;
            msg << "Incorrect method " << m;
            throw runtime_error(msg.str());
        }
    }
    else if (arg == PARAM_OFFSET)
    {
        ++i;

        if (i >= argc)
        {
            ostringstream msg;
            msg << "Missing value after " << PARAM_OFFSET;
            throw runtime_error(msg.str());
        }

        the_same_video_offset = atoi(argv[i]);
    }
    else
        return false;

    return true;
}

void App::printHelp()
{
    cout << "Demonstrates different optical flow algorithms.\n\n" 
         << "demo_optical_flow <frames_source1> [<frames_source2>] [flags]\n\n"
         << "Flags:\n"
         << "  " << PARAM_METHOD << " (brox|lk_dense|lk_sparse_gpu|lk_sparse_cpu|farneback_gpu|farneback_cpu)\n"
         << "  " << PARAM_OFFSET << " <integer>\n"
         << "      Frames offset for the duplicate video source in case when ony one video\n"
         << "      or camera source is given. The default is 1.\n";
    BaseApp::printHelp();
}

#ifdef USE_OPENGL
    void drawCallback(void* userdata)
    {
        DrawData* data = static_cast<DrawData*>(userdata);

        if (data->tex.empty() || data->arr.empty())
            return;

        static GlCamera camera;
        static bool init_camera = true;

        if (init_camera)
        {
            camera.setOrthoProjection(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);
            camera.lookAt(Point3d(0.0, 0.0, 1.0), Point3d(0.0, 0.0, 0.0), Point3d(0.0, 1.0, 0.0));
            init_camera = false;
        }

        camera.setupProjectionMatrix();
        camera.setupModelViewMatrix();

        render(data->tex);
        render(data->arr, RenderMode::TRIANGLES);

        render(data->method, GlFont::get("Courier New", 24, GlFont::WEIGHT_BOLD), Scalar::all(255), Point2d(3.0, 0.0));
        render(data->total_fps, GlFont::get("Courier New", 24, GlFont::WEIGHT_BOLD), Scalar::all(255), Point2d(3.0, 30.0));
        render(data->proc_fps, GlFont::get("Courier New", 24, GlFont::WEIGHT_BOLD), Scalar::all(255), Point2d(3.0, 60.0));
    }
#endif

void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

void drawArrowsDense(const Mat& u, const Mat& v, Mat& frame, Scalar line_color = Scalar(0, 0, 255))
{
    const int NEEDLE_MAP_SCALE = 16;
    const int line_thickness = 1;
    const float scale = 1.0f / (NEEDLE_MAP_SCALE * NEEDLE_MAP_SCALE);

    static Mat u_sum;
    static Mat v_sum;

    integral(u, u_sum, CV_32F);
    integral(v, v_sum, CV_32F);

    for (int y = NEEDLE_MAP_SCALE / 2; y < u.rows- NEEDLE_MAP_SCALE / 2; y += NEEDLE_MAP_SCALE)
    {
        for (int x = NEEDLE_MAP_SCALE / 2; x < u.cols- NEEDLE_MAP_SCALE / 2; x += NEEDLE_MAP_SCALE)
        {
            float u_avg = u_sum.at<float>(y - NEEDLE_MAP_SCALE / 2, x - NEEDLE_MAP_SCALE / 2) + 
                          u_sum.at<float>(y + NEEDLE_MAP_SCALE / 2, x + NEEDLE_MAP_SCALE / 2) -
                          u_sum.at<float>(y + NEEDLE_MAP_SCALE / 2, x - NEEDLE_MAP_SCALE / 2) -
                          u_sum.at<float>(y - NEEDLE_MAP_SCALE / 2, x + NEEDLE_MAP_SCALE / 2);

            float v_avg = v_sum.at<float>(y - NEEDLE_MAP_SCALE / 2, x - NEEDLE_MAP_SCALE / 2) + 
                          v_sum.at<float>(y + NEEDLE_MAP_SCALE / 2, x + NEEDLE_MAP_SCALE / 2) -
                          v_sum.at<float>(y + NEEDLE_MAP_SCALE / 2, x - NEEDLE_MAP_SCALE / 2) -
                          v_sum.at<float>(y - NEEDLE_MAP_SCALE / 2, x + NEEDLE_MAP_SCALE / 2);

            u_avg *= scale;
            v_avg *= scale;

            Point p(x, y);
            Point q(x + u_avg, y + v_avg);

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

void App::process()
{
    Ptr<PairFrameSource> source;

    if (sources.size() == 1)
        source = PairFrameSource::get(sources[0], the_same_video_offset);
    else if (sources.size() == 2)
        source = PairFrameSource::get(sources[0], sources[1]);
    else
    {
        cout << "Using default frame sources...\n";
        sources.resize(2);
        sources[0] = new ImageSource("data/optical_flow/rubberwhale1.png");
        sources[1] = new ImageSource("data/optical_flow/rubberwhale2.png");

        source = PairFrameSource::get(sources[0], sources[1]);
    }

#ifdef USE_OPENGL
    bool useGl = false;

    try
    {
        namedWindow("optical_flow_demo", WINDOW_OPENGL);
        setGlDevice();
        useGl = true;
    }
    catch (const cv::Exception&)
    {
        namedWindow("optical_flow_demo");
        useGl = false;
    }
#endif

    Mat frame0, frame1, uv, u, v;

    Mat frame0Gray, frame1Gray;

    GpuMat d_frame0Color, d_frame1Color;
    GpuMat d_frame0Gray, d_frame1Gray;
    GpuMat d_frame0Color32F, d_frame1Color32F;
    GpuMat d_frame0Gray32F, d_frame1Gray32F;

    GpuMat d_uv, d_u, d_v;

    GpuMat d_prevPts;
    GpuMat d_nextPts;
    GpuMat d_status;

    vector<Point2f> prevPts;
    vector<Point2f> nextPts;
    vector<uchar> status;

    GpuMat d_vertex, d_colors;

    int64 proc_start;
    double total_fps = 0;

    BroxOpticalFlow brox(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/, 10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);
    PyrLKOpticalFlow lk;
    GoodFeaturesToTrackDetector_GPU detector(8000, 0.01, 0.0);
    FarnebackOpticalFlow farneback_gpu; 

    while (!exited)
    {
        int64 start = getTickCount();

        source->next(frame0, frame1);

        d_frame0Color.upload(frame0);
        d_frame1Color.upload(frame1);

        if (method == BROX)
        {
            d_frame0Color.convertTo(d_frame0Color32F, CV_32F, 1.0 / 255.0);
            d_frame1Color.convertTo(d_frame1Color32F, CV_32F, 1.0 / 255.0);

            cvtColor(d_frame0Color32F, d_frame0Gray32F, COLOR_BGR2GRAY);
            cvtColor(d_frame1Color32F, d_frame1Gray32F, COLOR_BGR2GRAY);

            proc_start = getTickCount();

            brox(d_frame0Gray32F, d_frame1Gray32F, d_u, d_v);

            double proc_fps = getTickFrequency() / (getTickCount() - proc_start);

#ifdef USE_OPENGL
            if (useGl)
            {
                createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);

                stringstream total_fps_str;
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str;
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                draw_data.tex.copyFrom(d_frame0Gray32F);
                draw_data.arr.setVertexArray(d_vertex);
                draw_data.arr.setColorArray(d_colors, false);
                draw_data.method = "Brox GPU";
                draw_data.total_fps = total_fps_str.str();
                draw_data.proc_fps = proc_fps_str.str();
                setOpenGlDrawCallback("optical_flow_demo", drawCallback, &draw_data);
                updateWindow("optical_flow_demo");
            }
            else
#endif
            {
                Mat image = frame0.clone();
                d_u.download(u);
                d_v.download(v);

                drawArrowsDense(u, v, image, Scalar(255, 0, 0));

                stringstream total_fps_str; 
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str; 
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                printText(image, "Brox GPU", 0);
                printText(image, total_fps_str.str(), 1);
                printText(image, proc_fps_str.str(), 2);

                imshow("optical_flow_demo", image);
            }
        }
        else if (method == LK_DENSE)
        {
            cvtColor(d_frame0Color, d_frame0Gray, COLOR_BGR2GRAY);
            cvtColor(d_frame1Color, d_frame1Gray, COLOR_BGR2GRAY);

            proc_start = getTickCount();

            lk.dense(d_frame0Gray, d_frame1Gray, d_u, d_v);

            double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

#ifdef USE_OPENGL
            if (useGl)
            {
                createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);

                stringstream total_fps_str;
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str;
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                draw_data.tex.copyFrom(d_frame0Gray);
                draw_data.arr.setVertexArray(d_vertex);
                draw_data.arr.setColorArray(d_colors, false);
                draw_data.method = "PyrLK Dense GPU";
                draw_data.total_fps = total_fps_str.str();
                draw_data.proc_fps = proc_fps_str.str();
                setOpenGlDrawCallback("optical_flow_demo", drawCallback, &draw_data);
                updateWindow("optical_flow_demo");
            }
            else
#endif
            {
                Mat image = frame0.clone();
                d_u.download(u);
                d_v.download(v);

                drawArrowsDense(u, v, image, Scalar(255, 0, 0));

                stringstream total_fps_str; 
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str; 
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                printText(image, "PyrLK Dense GPU", 0);
                printText(image, total_fps_str.str(), 1);
                printText(image, proc_fps_str.str(), 2);

                imshow("optical_flow_demo", image);
            }
        }
        else if (method == LK_SPARSE_GPU)
        {
            cvtColor(d_frame0Color, d_frame0Gray, COLOR_BGR2GRAY);

            proc_start = getTickCount();

            detector(d_frame0Gray, d_prevPts);

            lk.sparse(d_frame0Color, d_frame1Color, d_prevPts, d_nextPts, d_status);

            double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

            download(d_prevPts, prevPts);
            download(d_nextPts, nextPts);
            download(d_status, status);

            Mat image = frame0.clone();

            drawArrows(image, prevPts, nextPts, status, Scalar(255, 0, 0));

            stringstream total_fps_str; 
            total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

            stringstream proc_fps_str; 
            proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

            printText(image, "PyrLK Sparse GPU", 0);
            printText(image, total_fps_str.str(), 1);
            printText(image, proc_fps_str.str(), 2);

            imshow("optical_flow_demo", image);
        }
        else if (method == LK_SPARSE_CPU)
        {
            cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);

            proc_start = getTickCount();

            goodFeaturesToTrack(frame0Gray, prevPts, 8000, 0.01, 0.0);

            calcOpticalFlowPyrLK(frame0, frame1, prevPts, nextPts, status, noArray());

            double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

            Mat image = frame0.clone();

            drawArrows(image, prevPts, nextPts, status, Scalar(255, 0, 0));

            stringstream total_fps_str; 
            total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

            stringstream proc_fps_str; 
            proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

            printText(image, "PyrLK Sparse CPU", 0);
            printText(image, total_fps_str.str(), 1);
            printText(image, proc_fps_str.str(), 2);

            imshow("optical_flow_demo", image);
        }
        else if (method == FARNEBACK_GPU)
        {
            cvtColor(d_frame0Color, d_frame0Gray, COLOR_BGR2GRAY);
            cvtColor(d_frame1Color, d_frame1Gray, COLOR_BGR2GRAY);

            proc_start = getTickCount();
            farneback_gpu(d_frame0Gray, d_frame1Gray, d_u, d_v);
            double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

#ifdef USE_OPENGL
            if (useGl)
            {
                createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);

                stringstream total_fps_str;
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str;
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                draw_data.tex.copyFrom(d_frame0Gray);
                draw_data.arr.setVertexArray(d_vertex);
                draw_data.arr.setColorArray(d_colors, false);
                draw_data.method = "Farneback GPU";
                draw_data.total_fps = total_fps_str.str();
                draw_data.proc_fps = proc_fps_str.str();
                setOpenGlDrawCallback("optical_flow_demo", drawCallback, &draw_data);
                updateWindow("optical_flow_demo");
            }
            else
#endif
            {
                Mat image = frame0.clone();
                d_u.download(u);
                d_v.download(v);

                drawArrowsDense(u, v, image, Scalar(255, 0, 0));

                stringstream total_fps_str; 
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str; 
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                printText(image, "Farneback GPU", 0);
                printText(image, total_fps_str.str(), 1);
                printText(image, proc_fps_str.str(), 2);

                imshow("optical_flow_demo", image);
            }
        }
        else if (method == FARNEBACK_CPU)
        {
            cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
            d_frame0Gray.upload(frame0Gray);
            cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);

            proc_start = getTickCount();
            calcOpticalFlowFarneback(
                    frame0Gray, frame1Gray, uv, farneback_gpu.pyrScale, farneback_gpu.numLevels, farneback_gpu.winSize, 
                    farneback_gpu.numIters, farneback_gpu.polyN, farneback_gpu.polySigma, farneback_gpu.flags);
            double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

#ifdef USE_OPENGL
            if (useGl)
            {
                d_uv.upload(uv);
                GpuMat uv_planes[] = {d_u, d_v};
                split(d_uv, uv_planes);

                createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);

                stringstream total_fps_str;
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str;
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                draw_data.tex.copyFrom(d_frame0Gray);
                draw_data.arr.setVertexArray(d_vertex);
                draw_data.arr.setColorArray(d_colors, false);
                draw_data.method = "Farneback CPU";
                draw_data.total_fps = total_fps_str.str();
                draw_data.proc_fps = proc_fps_str.str();
                setOpenGlDrawCallback("optical_flow_demo", drawCallback, &draw_data);
                updateWindow("optical_flow_demo");
            }
            else
#endif
            {
                Mat uv_planes[] = {u, v};
                split(uv, uv_planes);

                Mat image = frame0.clone();
                d_u.download(u);
                d_v.download(v);

                drawArrowsDense(u, v, image, Scalar(255, 0, 0));

                stringstream total_fps_str; 
                total_fps_str << "Total FPS : " << setprecision(4) << total_fps;

                stringstream proc_fps_str; 
                proc_fps_str << "Processing FPS : " << setprecision(4) << proc_fps;

                printText(image, "Farneback CPU", 0);
                printText(image, total_fps_str.str(), 1);
                printText(image, proc_fps_str.str(), 2);

                imshow("optical_flow_demo", image);
            }
        }

        processKey(waitKey(10));

        total_fps = getTickFrequency()  / (getTickCount() - proc_start);
    }
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        if (method == BROX)
            method = LK_DENSE;
        else if (method == LK_DENSE)
            method = LK_SPARSE_GPU;
        else if (method == LK_SPARSE_GPU)
            method = LK_SPARSE_CPU;
        else if (method == LK_SPARSE_CPU)
            method = FARNEBACK_GPU;
        else if (method == FARNEBACK_GPU)
            method = FARNEBACK_CPU;
        else if (method == FARNEBACK_CPU)
            method = BROX;
        break;
    }

    return false;
}

RUN_APP(App)

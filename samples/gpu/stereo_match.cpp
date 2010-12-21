#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


struct Params
{
    Params();
    static Params read(int argc, char** argv);

    string left;
    string right;
    string method_str;
    enum {BM, BP, CSBP} method;
    int ndisp; // Max disparity + 1
};


struct App
{
    App(const Params& p);
    void run();
    void handleKey(char key);

    void workBegin() { work_begin = getTickCount(); }
    void workEnd() 
    {
        int64 d = getTickCount() - work_begin;
        double f = getTickFrequency();
        work_fps = f / d;
    }

    string text() const
    {
        stringstream ss;
        ss << "(" << p.method_str << ") FPS: " << setiosflags(ios::left) << setprecision(4) << work_fps;
        return ss.str();
    }
private:
    Params p;
    bool running;

    gpu::StereoBM_GPU bm;
    gpu::StereoBeliefPropagation bp;
    gpu::StereoConstantSpaceBP csbp;

    int64 work_begin;
    double work_fps;
};


int main(int argc, char** argv)
{
    try
    {
        if (argc < 2)
        {
            cout << "Usage: stereo_match_gpu\n"
                << "\t-l <left_view> -r <right_view> # must be rectified\n"
                << "\t-m <stereo_match_method> # bm | bp | csbp\n";
            return 1;
        }
        App app(Params::read(argc, argv));
        app.run();
    }
    catch (const exception& e)
    {
        cout << "error: " << e.what() << endl;
    }
    return 0;
}


Params::Params()
{
    ndisp = 64;
}


Params Params::read(int argc, char** argv)
{
    Params p;

    for (int i = 1; i < argc - 1; i += 2)
    {
        string key = argv[i];
        string val = argv[i + 1];
        if (key == "-l") p.left = val;
        else if (key == "-r") p.right = val;
        else if (key == "-m") 
        {
            if (val == "BM") p.method = BM;
            else if (val == "BP") p.method = BP;
            else if (val == "CSBP") p.method = CSBP;
            else throw runtime_error("unknown stereo match method: " + val);
            p.method_str = val;
        }
        else if (key == "-ndisp") p.ndisp = atoi(val.c_str());
        else throw runtime_error("unknown key: " + key);
    }

    return p;
}


App::App(const Params& p)
    : p(p), running(false) 
{
    cout << "stereo_match_gpu sample\n";
    cout << "\nControls:\n"
        << "\tesc - exit\n"
        << "\tm - change stereo match method\n"
        << "\t1/q - increase/decrease max disprity\n";
}


void App::run()
{
    Mat left, right;
    Mat left_aux, right_aux;
    gpu::GpuMat d_left, d_right;

    // Load images
    left_aux = imread(p.left);
    right_aux = imread(p.right);
    if (left_aux.empty()) throw runtime_error("can't open file \"" + p.left + "\"");
    if (right_aux.empty()) throw runtime_error("can't open file \"" + p.right + "\"");
    cvtColor(left_aux, left, CV_BGR2GRAY);
    cvtColor(right_aux, right, CV_BGR2GRAY);
    d_left = left;
    d_right = right;

    imshow("left", left);
    imshow("right", right);

    // Create stero method descriptors
    bm.ndisp = p.ndisp;
    bp.ndisp = p.ndisp;
    csbp.ndisp = p.ndisp;

    // Prepare disparity map of specified type
    Mat disp(left.size(), CV_8U);
    gpu::GpuMat d_disp(left.size(), CV_8U);

    // Show initial parameters
    cout << "\nInitial Params:\n"
        << "\timage_size: (" << left.cols << ", " << left.rows << ")\n"
        << "\tmethod: " << p.method_str << endl
        << "\tndisp: " << p.ndisp << endl << endl;

    running = true;
    while (running)
    {
        workBegin();
        switch (p.method)
        {
        case Params::BM: bm(d_left, d_right, d_disp); break;
        case Params::BP: bp(d_left, d_right, d_disp); break;
        case Params::CSBP: csbp(d_left, d_right, d_disp); break;
        }
        workEnd();

        // Show results
        disp = d_disp;
        putText(disp, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
        imshow("disparity", disp);

        handleKey((char)waitKey(3));
    }
}


void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm': case 'M':
        switch (p.method)
        {
        case Params::BM:
            p.method = Params::BP;
            p.method_str = "BP";
            break;
        case Params::BP:
            p.method = Params::CSBP;
            p.method_str = "CSBP";
            break;
        case Params::CSBP:
            p.method = Params::BM;
            p.method_str = "BM";
            break;
        }
        cout << "method: " << p.method_str << endl;
        break;
    case '1':
        p.ndisp = p.ndisp == 1 ? 8 : p.ndisp + 8;
        bm.ndisp = p.ndisp;
        bp.ndisp = p.ndisp;
        csbp.ndisp = p.ndisp;
        cout << "ndisp: " << p.ndisp << endl;
        break;
    case 'q': case 'Q':
        p.ndisp = max(p.ndisp - 8, 1);
        bm.ndisp = p.ndisp;
        bp.ndisp = p.ndisp;
        csbp.ndisp = p.ndisp;
        cout << "ndisp: " << p.ndisp << endl;
        break;
    }
}


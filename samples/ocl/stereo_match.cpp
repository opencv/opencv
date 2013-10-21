#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "opencv2/core/utility.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace ocl;


struct App
{
    App(CommandLineParser& cmd);
    void run();
    void handleKey(char key);
    void printParams() const;

    void workBegin()
    {
        work_begin = getTickCount();
    }
    void workEnd()
    {
        int64 d = getTickCount() - work_begin;
        double f = getTickFrequency();
        work_fps = f / d;
    }
    string method_str() const
    {
        switch (method)
        {
        case BM:
            return "BM";
        case BP:
            return "BP";
        case CSBP:
            return "CSBP";
        }
        return "";
    }
    string text() const
    {
        stringstream ss;
        ss << "(" << method_str() << ") FPS: " << setiosflags(ios::left)
           << setprecision(4) << work_fps;
        return ss.str();
    }
private:
    bool running, write_once;

    Mat left_src, right_src;
    Mat left, right;
    oclMat d_left, d_right;

    StereoBM_OCL bm;
    StereoBeliefPropagation bp;
    StereoConstantSpaceBP csbp;

    int64 work_begin;
    double work_fps;

    string l_img, r_img;
    string out_img;
    enum {BM, BP, CSBP} method;
    int ndisp; // Max disparity + 1
    enum {GPU, CPU} type;
};

int main(int argc, char** argv)
{
    const char* keys =
        "{ h | help     | false                     | print help message }"
        "{ l | left     |                           | specify left image }"
        "{ r | right    |                           | specify right image }"
        "{ m | method   | BM                        | specify match method(BM/BP/CSBP) }"
        "{ n | ndisp    | 64                        |  specify number of disparity levels }"
        "{ o | output   | stereo_match_output.jpg   | specify output path when input is images}";
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Available options:" << endl;
        cmd.printMessage();
        return 0;
    }
    try
    {
        App app(cmd);
        cout << "Device name:" << cv::ocl::Context::getContext()->getDeviceInfo().deviceName << endl;

        app.run();
    }
    catch (const exception& e)
    {
        cout << "error: " << e.what() << endl;
    }
    return 0;
}

App::App(CommandLineParser& cmd)
    : running(false),method(BM)
{
    cout << "stereo_match_ocl sample\n";
    cout << "\nControls:\n"
         << "\tesc - exit\n"
         << "\to - save output image once\n"
         << "\tp - print current parameters\n"
         << "\tg - convert source images into gray\n"
         << "\tm - change stereo match method\n"
         << "\ts - change Sobel prefiltering flag (for BM only)\n"
         << "\t1/q - increase/decrease maximum disparity\n"
         << "\t2/w - increase/decrease window size (for BM only)\n"
         << "\t3/e - increase/decrease iteration count (for BP and CSBP only)\n"
         << "\t4/r - increase/decrease level count (for BP and CSBP only)\n";
    l_img = cmd.get<string>("l");
    r_img = cmd.get<string>("r");
    string mstr = cmd.get<string>("m");
    if(mstr == "BM") method = BM;
    else if(mstr == "BP") method = BP;
    else if(mstr == "CSBP") method = CSBP;
    else cout << "unknown method!\n";
    ndisp = cmd.get<int>("n");
    out_img = cmd.get<string>("o");
    write_once = false;
}


void App::run()
{
    // Load images
    left_src = imread(l_img);
    right_src = imread(r_img);
    if (left_src.empty()) throw runtime_error("can't open file \"" + l_img + "\"");
    if (right_src.empty()) throw runtime_error("can't open file \"" + r_img + "\"");

    cvtColor(left_src, left, COLOR_BGR2GRAY);
    cvtColor(right_src, right, COLOR_BGR2GRAY);

    d_left.upload(left);
    d_right.upload(right);

    imshow("left", left);
    imshow("right", right);

    // Set common parameters
    bm.ndisp = ndisp;
    bp.ndisp = ndisp;
    csbp.ndisp = ndisp;

    cout << endl;
    printParams();

    running = true;
    while (running)
    {
        // Prepare disparity map of specified type
        Mat disp;
        oclMat d_disp;
        workBegin();
        switch (method)
        {
        case BM:
            if (d_left.channels() > 1 || d_right.channels() > 1)
            {
                cout << "BM doesn't support color images\n";
                cvtColor(left_src, left, COLOR_BGR2GRAY);
                cvtColor(right_src, right, COLOR_BGR2GRAY);
                cout << "image_channels: " << left.channels() << endl;
                d_left.upload(left);
                d_right.upload(right);
                imshow("left", left);
                imshow("right", right);
            }
            bm(d_left, d_right, d_disp);
            break;
        case BP:
            bp(d_left, d_right, d_disp);
            break;
        case CSBP:
            csbp(d_left, d_right, d_disp);
            break;
        }

        // Show results
        d_disp.download(disp);
        workEnd();

        if (method != BM)
        {
            disp.convertTo(disp, 0);
        }
        putText(disp, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
        imshow("disparity", disp);
        if(write_once)
        {
            imwrite(out_img, disp);
            write_once = false;
        }
        handleKey((char)waitKey(3));
    }
}


void App::printParams() const
{
    cout << "--- Parameters ---\n";
    cout << "image_size: (" << left.cols << ", " << left.rows << ")\n";
    cout << "image_channels: " << left.channels() << endl;
    cout << "method: " << method_str() << endl
         << "ndisp: " << ndisp << endl;
    switch (method)
    {
    case BM:
        cout << "win_size: " << bm.winSize << endl;
        cout << "prefilter_sobel: " << bm.preset << endl;
        break;
    case BP:
        cout << "iter_count: " << bp.iters << endl;
        cout << "level_count: " << bp.levels << endl;
        break;
    case CSBP:
        cout << "iter_count: " << csbp.iters << endl;
        cout << "level_count: " << csbp.levels << endl;
        break;
    }
    cout << endl;
}


void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'p':
    case 'P':
        printParams();
        break;
    case 'g':
    case 'G':
        if (left.channels() == 1 && method != BM)
        {
            left = left_src;
            right = right_src;
        }
        else
        {
            cvtColor(left_src, left, COLOR_BGR2GRAY);
            cvtColor(right_src, right, COLOR_BGR2GRAY);
        }
        d_left.upload(left);
        d_right.upload(right);
        cout << "image_channels: " << left.channels() << endl;
        imshow("left", left);
        imshow("right", right);
        break;
    case 'm':
    case 'M':
        switch (method)
        {
        case BM:
            method = BP;
            break;
        case BP:
            method = CSBP;
            break;
        case CSBP:
            method = BM;
            break;
        }
        cout << "method: " << method_str() << endl;
        break;
    case 's':
    case 'S':
        if (method == BM)
        {
            switch (bm.preset)
            {
            case StereoBM_OCL::BASIC_PRESET:
                bm.preset = StereoBM_OCL::PREFILTER_XSOBEL;
                break;
            case StereoBM_OCL::PREFILTER_XSOBEL:
                bm.preset = StereoBM_OCL::BASIC_PRESET;
                break;
            }
            cout << "prefilter_sobel: " << bm.preset << endl;
        }
        break;
    case '1':
        ndisp == 1 ? ndisp = 8 : ndisp += 8;
        cout << "ndisp: " << ndisp << endl;
        bm.ndisp = ndisp;
        bp.ndisp = ndisp;
        csbp.ndisp = ndisp;
        break;
    case 'q':
    case 'Q':
        ndisp = max(ndisp - 8, 1);
        cout << "ndisp: " << ndisp << endl;
        bm.ndisp = ndisp;
        bp.ndisp = ndisp;
        csbp.ndisp = ndisp;
        break;
    case '2':
        if (method == BM)
        {
            bm.winSize = min(bm.winSize + 1, 51);
            cout << "win_size: " << bm.winSize << endl;
        }
        break;
    case 'w':
    case 'W':
        if (method == BM)
        {
            bm.winSize = max(bm.winSize - 1, 2);
            cout << "win_size: " << bm.winSize << endl;
        }
        break;
    case '3':
        if (method == BP)
        {
            bp.iters += 1;
            cout << "iter_count: " << bp.iters << endl;
        }
        else if (method == CSBP)
        {
            csbp.iters += 1;
            cout << "iter_count: " << csbp.iters << endl;
        }
        break;
    case 'e':
    case 'E':
        if (method == BP)
        {
            bp.iters = max(bp.iters - 1, 1);
            cout << "iter_count: " << bp.iters << endl;
        }
        else if (method == CSBP)
        {
            csbp.iters = max(csbp.iters - 1, 1);
            cout << "iter_count: " << csbp.iters << endl;
        }
        break;
    case '4':
        if (method == BP)
        {
            bp.levels += 1;
            cout << "level_count: " << bp.levels << endl;
        }
        else if (method == CSBP)
        {
            csbp.levels += 1;
            cout << "level_count: " << csbp.levels << endl;
        }
        break;
    case 'r':
    case 'R':
        if (method == BP)
        {
            bp.levels = max(bp.levels - 1, 1);
            cout << "level_count: " << bp.levels << endl;
        }
        else if (method == CSBP)
        {
            csbp.levels = max(csbp.levels - 1, 1);
            cout << "level_count: " << csbp.levels << endl;
        }
        break;
    case 'o':
    case 'O':
        write_once = true;
        break;
    }
}

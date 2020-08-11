/*
 * The example of interoperability between SYCL/OpenCL and OpenCV.
 * - SYCL: https://www.khronos.org/sycl/
 * - SYCL runtime parameters: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
 */
#include <CL/sycl.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/ocl.hpp>


class sycl_inverse_kernel;  // can be omitted - modern SYCL versions doesn't require this

using namespace cv;


class App
{
public:
    App(const CommandLineParser& cmd);
    ~App();

    void initVideoSource();

    void initSYCL();

    void process_frame(cv::Mat& frame);

    /// to check result with CPU-only reference code
    Mat process_frame_reference(const cv::Mat& frame);

    int run();

    bool isRunning() { return m_running; }
    bool doProcess() { return m_process; }

    void setRunning(bool running)      { m_running = running; }
    void setDoProcess(bool process)    { m_process = process; }

protected:
    void handleKey(char key);

private:
    bool                        m_running;
    bool                        m_process;
    bool                        m_show_ui;

    int64                       m_t0;
    int64                       m_t1;
    float                       m_time;
    float                       m_frequency;

    std::string                 m_file_name;
    int                         m_camera_id;
    cv::VideoCapture            m_cap;
    cv::Mat                     m_frame;

    cl::sycl::queue sycl_queue;
};


App::App(const CommandLineParser& cmd)
{
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<std::string>("video");

    m_running    = false;
    m_process    = false;
} // ctor


App::~App()
{
    // nothing
}


void App::initSYCL()
{
    using namespace cl::sycl;

    // Configuration details: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
    cl::sycl::default_selector selector;

    sycl_queue = cl::sycl::queue(selector, [](cl::sycl::exception_list l)
    {
        // exception_handler
        for (auto ep : l)
        {
            try
            {
                std::rethrow_exception(ep);
            }
            catch (const cl::sycl::exception& e)
            {
                std::cerr << "SYCL exception: " << e.what() << std::endl;
            }
        }
    });

    auto device = sycl_queue.get_device();
    auto platform = device.get_platform();
    std::cout << "SYCL device: " << device.get_info<info::device::name>()
        << " @ " << device.get_info<info::device::driver_version>()
        << " (platform: " << platform.get_info<info::platform::name>() << ")" << std::endl;

    if (device.is_host())
    {
        std::cerr << "SYCL can't select OpenCL device. Host is used for computations, interoperability is not available" << std::endl;
    }
    else
    {
        // bind OpenCL context/device/queue from SYCL to OpenCV
        try
        {
            auto ctx = cv::ocl::OpenCLExecutionContext::create(
                    platform.get_info<info::platform::name>(),
                    platform.get(),
                    sycl_queue.get_context().get(),
                    device.get()
                );
            ctx.bind();
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "OpenCV: Can't bind SYCL OpenCL context/device/queue: " << e.what() << std::endl;
        }
        std::cout << "OpenCV uses OpenCL: " << (cv::ocl::useOpenCL() ? "True" : "False") << std::endl;
    }
} // initSYCL()


void App::initVideoSource()
{
    if (!m_file_name.empty() && m_camera_id == -1)
    {
        m_cap.open(samples::findFileOrKeep(m_file_name));
        if (!m_cap.isOpened())
            throw std::runtime_error(std::string("can't open video stream: ") + m_file_name);
    }
    else if (m_camera_id != -1)
    {
        m_cap.open(m_camera_id);
        if (!m_cap.isOpened())
            throw std::runtime_error(std::string("can't open camera: ") + std::to_string(m_camera_id));
    }
    else
        throw std::runtime_error(std::string("specify video source"));
} // initVideoSource()


void App::process_frame(cv::Mat& frame)
{
    using namespace cl::sycl;

    // cv::Mat => cl::sycl::buffer
    {
        CV_Assert(frame.isContinuous());
        CV_CheckTypeEQ(frame.type(), CV_8UC1, "");

        buffer<uint8_t, 2> frame_buffer(frame.data, range<2>(frame.rows, frame.cols));

        // done automatically: frame_buffer.set_write_back(true);

        sycl_queue.submit([&](handler& cgh) {
          auto pixels = frame_buffer.get_access<access::mode::read_write>(cgh);

          cgh.parallel_for<class sycl_inverse_kernel>(range<2>(frame.rows, frame.cols), [=](item<2> item) {
              uint8_t v = pixels[item];
              pixels[item] = ~v;
          });
        });

        sycl_queue.wait_and_throw();
    }

    // No way to extract cl_mem from cl::sycl::buffer (ref: 3.6.11 "Interfacing with OpenCL" of SYCL 1.2.1)
    // We just reusing OpenCL context/device/queue from SYCL here (see initSYCL() bind part) and call UMat processing
    {
        UMat blurResult;
        {
            UMat umat_buffer = frame.getUMat(ACCESS_RW);
            cv::blur(umat_buffer, blurResult, Size(3, 3));  // UMat doesn't support inplace
        }
        Mat result;
        blurResult.copyTo(result);
        swap(result, frame);
    }
}

Mat App::process_frame_reference(const cv::Mat& frame)
{
    Mat result;
    cv::bitwise_not(frame, result);
    Mat blurResult;
    cv::blur(result, blurResult, Size(3, 3));  // avoid inplace
    blurResult.copyTo(result);
    return result;
}

int App::run()
{
    std::cout << "Initializing..." << std::endl;

    initSYCL();
    initVideoSource();

    std::cout << "Press ESC to exit" << std::endl;
    std::cout << "      'p' to toggle ON/OFF processing" << std::endl;

    m_running = true;
    m_process = true;
    m_show_ui = true;

    int processedFrames = 0;

    cv::TickMeter timer;

    // Iterate over all frames
    while (isRunning() && m_cap.read(m_frame))
    {
        Mat m_frameGray;
        cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

        bool checkWithReference = (processedFrames == 0);
        Mat reference_result;
        if (checkWithReference)
        {
            reference_result = process_frame_reference(m_frameGray);
        }

        timer.reset();
        timer.start();

        if (m_process)
        {
            process_frame(m_frameGray);
        }

        timer.stop();

        if (checkWithReference)
        {
            double diffInf = cv::norm(reference_result, m_frameGray, NORM_INF);
            if (diffInf > 0)
            {
                std::cerr << "Result is not accurate. diffInf=" << diffInf << std::endl;
                imwrite("reference.png", reference_result);
                imwrite("actual.png", m_frameGray);
            }
        }

        Mat img_to_show = m_frameGray;

        std::ostringstream msg;
        msg << "Frame " << processedFrames << " (" << m_frame.size
            << ")   Time: " << cv::format("%.2f", timer.getTimeMilli()) << " msec"
            << " (process: " << (m_process ? "True" : "False") << ")";
        std::cout << msg.str() << std::endl;
        putText(img_to_show, msg.str(), Point(5, 150), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);

        if (m_show_ui)
        {
            try
            {
                imshow("sycl_interop", img_to_show);
                int key = waitKey(1);
                switch (key)
                {
                case 27:  // ESC
                    m_running = false;
                    break;

                case 'p':  // fallthru
                case 'P':
                    m_process = !m_process;
                    break;

                default:
                    break;
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "ERROR(OpenCV UI): " << e.what() << std::endl;
                if (processedFrames > 0)
                    throw;
                m_show_ui = false;  // UI is not available
            }
        }

        processedFrames++;

        if (!m_show_ui)
        {
            if (processedFrames > 100)
                m_running = false;
        }
    }

    return 0;
}


int main(int argc, char** argv)
{
    const char* keys =
        "{ help h ?    |          | print help message }"
        "{ camera c    | -1       | use camera as input }"
        "{ video  v    |          | use video as input }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    try
    {
        App app(cmd);
        if (!cmd.check())
        {
            cmd.printErrors();
            return 1;
        }
        app.run();
    }
    catch (const cv::Exception& e)
    {
        std::cout << "FATAL: OpenCV error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cout << "FATAL: C++ error: " << e.what() << std::endl;
        return 1;
    }

    catch (...)
    {
        std::cout << "FATAL: unknown C++ exception" << std::endl;
        return 1;
    }

    return EXIT_SUCCESS;
} // main()

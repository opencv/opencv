#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include <CL/cl.h>

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

namespace opencl {

class PlatformInfo
{
public:
    PlatformInfo()
    {}

    ~PlatformInfo()
    {}

    cl_int QueryInfo(cl_platform_id id)
    {
        query_param(id, CL_PLATFORM_PROFILE, m_profile);
        query_param(id, CL_PLATFORM_VERSION, m_version);
        query_param(id, CL_PLATFORM_NAME, m_name);
        query_param(id, CL_PLATFORM_VENDOR, m_vendor);
        query_param(id, CL_PLATFORM_EXTENSIONS, m_extensions);
        return CL_SUCCESS;
    }

    std::string Profile()    { return m_profile; }
    std::string Version()    { return m_version; }
    std::string Name()       { return m_name; }
    std::string Vendor()     { return m_vendor; }
    std::string Extensions() { return m_extensions; }

private:
    cl_int query_param(cl_platform_id id, cl_platform_info param, std::string& paramStr)
    {
        cl_int res;

        size_t psize;
        cv::AutoBuffer<char> buf;

        res = clGetPlatformInfo(id, param, 0, 0, &psize);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetPlatformInfo failed"));

        buf.resize(psize);
        res = clGetPlatformInfo(id, param, psize, buf, 0);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetPlatformInfo failed"));

        // just in case, ensure trailing zero for ASCIIZ string
        buf[psize] = 0;

        paramStr = buf;

        return CL_SUCCESS;
    }

private:
    std::string m_profile;
    std::string m_version;
    std::string m_name;
    std::string m_vendor;
    std::string m_extensions;
};


class DeviceInfo
{
public:
    DeviceInfo()
    {}

    ~DeviceInfo()
    {}

    cl_int QueryInfo(cl_device_id id)
    {
        query_param(id, CL_DEVICE_TYPE, m_type);
        query_param(id, CL_DEVICE_VENDOR_ID, m_vendor_id);
        query_param(id, CL_DEVICE_MAX_COMPUTE_UNITS, m_max_compute_units);
        query_param(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, m_max_work_item_dimensions);
        query_param(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, m_max_work_item_sizes);
        query_param(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, m_max_work_group_size);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, m_preferred_vector_width_char);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, m_preferred_vector_width_short);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, m_preferred_vector_width_int);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, m_preferred_vector_width_long);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, m_preferred_vector_width_float);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, m_preferred_vector_width_double);
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, m_preferred_vector_width_half);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, m_native_vector_width_char);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, m_native_vector_width_short);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, m_native_vector_width_int);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, m_native_vector_width_long);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, m_native_vector_width_float);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, m_native_vector_width_double);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, m_native_vector_width_half);
        query_param(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, m_max_clock_frequency);
        query_param(id, CL_DEVICE_ADDRESS_BITS, m_address_bits);
        query_param(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, m_max_mem_alloc_size);
        query_param(id, CL_DEVICE_IMAGE_SUPPORT, m_image_support);
        query_param(id, CL_DEVICE_MAX_READ_IMAGE_ARGS, m_max_read_image_args);
        query_param(id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, m_max_write_image_args);
        query_param(id, CL_DEVICE_IMAGE2D_MAX_WIDTH, m_image2d_max_width);
        query_param(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, m_image2d_max_height);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_WIDTH, m_image3d_max_width);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, m_image3d_max_height);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_DEPTH, m_image3d_max_depth);
        query_param(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, m_image_max_buffer_size);
        query_param(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, m_image_max_array_size);
        query_param(id, CL_DEVICE_MAX_SAMPLERS, m_max_samplers);
        query_param(id, CL_DEVICE_MAX_PARAMETER_SIZE, m_max_parameter_size);
        query_param(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, m_mem_base_addr_align);
        query_param(id, CL_DEVICE_SINGLE_FP_CONFIG, m_single_fp_config);
        query_param(id, CL_DEVICE_DOUBLE_FP_CONFIG, m_double_fp_config);
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, m_global_mem_cache_type);
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, m_global_mem_cacheline_size);
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, m_global_mem_cache_size);
        query_param(id, CL_DEVICE_GLOBAL_MEM_SIZE, m_global_mem_size);
        query_param(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, m_max_constant_buffer_size);
        query_param(id, CL_DEVICE_MAX_CONSTANT_ARGS, m_max_constant_args);
        query_param(id, CL_DEVICE_LOCAL_MEM_TYPE, m_local_mem_type);
        query_param(id, CL_DEVICE_LOCAL_MEM_SIZE, m_local_mem_size);
        query_param(id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, m_error_correction_support);
#if defined(CL_VERSION_1_1)
        query_param(id, CL_DEVICE_HOST_UNIFIED_MEMORY, m_host_unified_memory);
#endif
        query_param(id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, m_profiling_timer_resolution);
        query_param(id, CL_DEVICE_ENDIAN_LITTLE, m_endian_little);
        query_param(id, CL_DEVICE_AVAILABLE, m_available);
        query_param(id, CL_DEVICE_COMPILER_AVAILABLE, m_compiler_available);
        query_param(id, CL_DEVICE_LINKER_AVAILABLE, m_linker_available);
        query_param(id, CL_DEVICE_EXECUTION_CAPABILITIES, m_execution_capabilities);
#if defined(CL_VERSION_1_0)
        query_param(id, CL_DEVICE_QUEUE_PROPERTIES, m_queue_properties);
#endif
        query_param(id, CL_DEVICE_BUILT_IN_KERNELS, m_built_in_kernels);
        query_param(id, CL_DEVICE_PLATFORM, m_platform);
        query_param(id, CL_DEVICE_NAME, m_name);
        query_param(id, CL_DEVICE_VENDOR, m_vendor);
        query_param(id, CL_DRIVER_VERSION, m_driver_version);
        query_param(id, CL_DEVICE_PROFILE, m_profile);
        query_param(id, CL_DEVICE_VERSION, m_version);
        query_param(id, CL_DEVICE_OPENCL_C_VERSION, m_opencl_c_version);
        query_param(id, CL_DEVICE_EXTENSIONS, m_extensions);
        query_param(id, CL_DEVICE_PRINTF_BUFFER_SIZE, m_printf_buffer_size);
        query_param(id, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, m_preferred_interop_user_sync);
        query_param(id, CL_DEVICE_PARENT_DEVICE, m_parent_device);
        query_param(id, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, m_partition_max_sub_devices);
        query_param(id, CL_DEVICE_PARTITION_PROPERTIES, m_partition_properties);
        query_param(id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, m_partition_affinity_domain);
        query_param(id, CL_DEVICE_PARTITION_TYPE, m_partition_type);
        query_param(id, CL_DEVICE_REFERENCE_COUNT, m_reference_count);
        query_param(id, CL_DEVICE_ADDRESS_BITS, m_address_bits);
        return CL_SUCCESS;
    }

    std::string Name() { return m_name; }

private:
    template<typename T>
    cl_int query_param(cl_device_id id, cl_device_info param, T& value)
    {
        cl_int res;
        size_t size;

        res = clGetDeviceInfo(id, param, 0, 0, &size);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        if (0 == size)
            return CL_SUCCESS;

        if (sizeof(T) != size)
            throw std::runtime_error(std::string("clGetDeviceInfo: param size mismatch"));

        res = clGetDeviceInfo(id, param, size, &value, 0);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        return CL_SUCCESS;
    }

    template<typename T>
    cl_int query_param(cl_device_id id, cl_device_info param, std::vector<T>& value)
    {
        cl_int res;
        size_t size;

        res = clGetDeviceInfo(id, param, 0, 0, &size);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        if (0 == size)
            return CL_SUCCESS;

        value.resize(size / sizeof(T));

        res = clGetDeviceInfo(id, param, size, &value[0], 0);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        return CL_SUCCESS;
    }

    cl_int query_param(cl_device_id id, cl_device_info param, std::string& value)
    {
        cl_int res;
        size_t size;

        res = clGetDeviceInfo(id, param, 0, 0, &size);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        value.resize(size + 1);

        res = clGetDeviceInfo(id, param, size, &value[0], 0);
        if (CL_SUCCESS != res)
            throw std::runtime_error(std::string("clGetDeviceInfo failed"));

        // just in case, ensure trailing zero for ASCIIZ string
        value[size] = 0;

        return CL_SUCCESS;
    }

private:
    cl_device_type                            m_type;
    cl_uint                                   m_vendor_id;
    cl_uint                                   m_max_compute_units;
    cl_uint                                   m_max_work_item_dimensions;
    std::vector<size_t>                       m_max_work_item_sizes;
    size_t                                    m_max_work_group_size;
    cl_uint                                   m_preferred_vector_width_char;
    cl_uint                                   m_preferred_vector_width_short;
    cl_uint                                   m_preferred_vector_width_int;
    cl_uint                                   m_preferred_vector_width_long;
    cl_uint                                   m_preferred_vector_width_float;
    cl_uint                                   m_preferred_vector_width_double;
    cl_uint                                   m_preferred_vector_width_half;
    cl_uint                                   m_native_vector_width_char;
    cl_uint                                   m_native_vector_width_short;
    cl_uint                                   m_native_vector_width_int;
    cl_uint                                   m_native_vector_width_long;
    cl_uint                                   m_native_vector_width_float;
    cl_uint                                   m_native_vector_width_double;
    cl_uint                                   m_native_vector_width_half;
    cl_uint                                   m_max_clock_frequency;
    cl_uint                                   m_address_bits;
    cl_ulong                                  m_max_mem_alloc_size;
    cl_bool                                   m_image_support;
    cl_uint                                   m_max_read_image_args;
    cl_uint                                   m_max_write_image_args;
    cl_uint                                   m_max_read_write_image_args;
    size_t                                    m_image2d_max_width;
    size_t                                    m_image2d_max_height;
    size_t                                    m_image3d_max_width;
    size_t                                    m_image3d_max_height;
    size_t                                    m_image3d_max_depth;
    size_t                                    m_image_max_buffer_size;
    size_t                                    m_image_max_array_size;
    cl_uint                                   m_max_samplers;
    cl_uint                                   m_image_pitch_alignment;
    cl_uint                                   m_image_base_address_alignment;
    cl_uint                                   m_max_pipe_args;
    cl_uint                                   m_pipe_max_active_reservations;
    cl_uint                                   m_pipe_max_packet_size;
    size_t                                    m_max_parameter_size;
    cl_uint                                   m_mem_base_addr_align;
    cl_device_fp_config                       m_single_fp_config;
    cl_device_fp_config                       m_double_fp_config;
    cl_device_mem_cache_type                  m_global_mem_cache_type;
    cl_uint                                   m_global_mem_cacheline_size;
    cl_ulong                                  m_global_mem_cache_size;
    cl_ulong                                  m_global_mem_size;
    cl_ulong                                  m_max_constant_buffer_size;
    cl_uint                                   m_max_constant_args;
    size_t                                    m_max_global_variable_size;
    size_t                                    m_global_variable_preferred_total_size;
    cl_device_local_mem_type                  m_local_mem_type;
    cl_ulong                                  m_local_mem_size;
    cl_bool                                   m_error_correction_support;
#if defined(CL_VERSION_1_1)
    cl_bool                                   m_host_unified_memory;
#endif
    size_t                                    m_profiling_timer_resolution;
    cl_bool                                   m_endian_little;
    cl_bool                                   m_available;
    cl_bool                                   m_compiler_available;
    cl_bool                                   m_linker_available;
    cl_device_exec_capabilities               m_execution_capabilities;
#if defined(CL_VERSION_1_0)
    cl_command_queue_properties               m_queue_properties;
#endif
    cl_command_queue_properties               m_queue_on_host_properties;
    cl_command_queue_properties               m_queue_on_device_properties;
    cl_uint                                   m_queue_on_device_preferred_size;
    cl_uint                                   m_queue_on_device_max_size;
    cl_uint                                   m_max_on_device_queues;
    cl_uint                                   m_max_on_device_events;
    std::string                               m_built_in_kernels;
    cl_platform_id                            m_platform;
    std::string                               m_name;
    std::string                               m_vendor;
    std::string                               m_driver_version;
    std::string                               m_profile;
    std::string                               m_version;
    std::string                               m_opencl_c_version;
    std::string                               m_extensions;
    size_t                                    m_printf_buffer_size;
    cl_bool                                   m_preferred_interop_user_sync;
    cl_device_id                              m_parent_device;
    cl_uint                                   m_partition_max_sub_devices;
    std::vector<cl_device_partition_property> m_partition_properties;
    cl_device_affinity_domain                 m_partition_affinity_domain;
    std::vector<cl_device_partition_property> m_partition_type;
    cl_uint                                   m_reference_count;
};


} // namespace opencl


class App
{
public:
    App(CommandLineParser& cmd);

    int initVideoSource();
    int initOpenCL();

    int process_frame_with_open_cl(cv::Mat& frame, cl_mem* cl_buffer);
    int process_cl_buffer_with_opencv(cl_mem buffer, size_t step, int rows, int cols, int type, cv::UMat& u);

    int run();

protected:
    bool nextFrame(cv::Mat& frame) { return m_cap.read(frame); }
    void handleKey(char key);
    void timerStart();
    void timerEnd();
    std::string fpsStr() const;
    std::string message() const;

private:
    // Args args;
    bool running;
    bool process;

    int64 m_t0;
    int64 m_t1;
    double m_fps;

    string file_name;
    int camera_id;
    cv::VideoCapture m_cap;
    cv::Mat m_frame;
    cv::Mat m_frameGray;

    opencl::PlatformInfo m_platformInfo;
    opencl::DeviceInfo   m_deviceInfo;
    std::vector<cl_platform_id> m_platform_ids;
    cl_context       m_context;
    cl_device_id     m_device_id;
    cl_command_queue m_queue;
    cl_kernel        m_kernel;
};


App::App(CommandLineParser& cmd)
{
    cout << "\nPress ESC to exit\n"
         << endl;

    camera_id = cmd.get<int>("camera");
    file_name = cmd.get<string>("video");

    process = true;
} // ctor


int App::initVideoSource()
{
    try
    {
        if (!file_name.empty() && camera_id == -1)
        {
            m_cap.open(file_name.c_str());
            if (!m_cap.isOpened())
                throw std::runtime_error(std::string("can't open video file: " + file_name));
        }
        else if (camera_id != -1)
        {
            m_cap.open(camera_id);
            if (!m_cap.isOpened())
            {
                std::stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw std::runtime_error(msg.str());
            }
        }
        else
            throw std::runtime_error(std::string("specify video source"));
    }

    catch (std::exception e)
    {
        cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


int App::initOpenCL()
{
    cl_int res = CL_SUCCESS;
    cl_uint num_entries = 0;

    res = clGetPlatformIDs(0, 0, &num_entries);
    if (CL_SUCCESS != res)
        return -1;

    m_platform_ids.resize(num_entries);

    res = clGetPlatformIDs(num_entries, &m_platform_ids[0], 0);
    if (CL_SUCCESS != res)
        return -1;

    unsigned int i;

    // create context from first platform with GPU device
    for (i = 0; i < m_platform_ids.size(); i++)
    {
        cl_context_properties props[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(m_platform_ids[i]),
            0
        };

        m_context = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, 0, 0, &res);
        if (0 == m_context || CL_SUCCESS != res)
            continue;

        res = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_device_id, 0);
        if (CL_SUCCESS != res)
            return -1;

        m_queue = clCreateCommandQueue(m_context, m_device_id, 0, &res);
        if (0 == m_queue || CL_SUCCESS != res)
            return -1;

        const char* kernelSrc =
            "__kernel "
            "void bitwise_inv_8uC1("
            "    __global unsigned char* pSrcDst,"
            "             int            srcDstStep,"
            "             int            rows,"
            "             int            cols)"
            "{"
            "    int x = get_global_id(0);"
            "    int y = get_global_id(1);"
            "        int idx = mad24(y, srcDstStep, mad24(x, 1, 0));"
            "            pSrcDst[idx] = ~pSrcDst[idx];"
            "}";

        size_t len = strlen(kernelSrc);
        cl_program program = clCreateProgramWithSource(m_context, 1, &kernelSrc, &len, &res);
        if (0 == program || CL_SUCCESS != res)
            return -1;

        res = clBuildProgram(program, 1, &m_device_id, 0, 0, 0);
        if (CL_SUCCESS != res)
            return -1;

        m_kernel = clCreateKernel(program, "bitwise_inv_8uC1", &res);
        if (0 == m_kernel || CL_SUCCESS != res)
            return -1;

        m_platformInfo.QueryInfo(m_platform_ids[i]);
        m_deviceInfo.QueryInfo(m_device_id);

        // attach OpenCL context to OpenCV
        cv::ocl::attachContext(m_platformInfo.Name(), m_platform_ids[i], m_context, m_device_id);

        break;
    }

    return m_context != 0 ? CL_SUCCESS : -1;
} // initOpenCL()


int App::process_frame_with_open_cl(cv::Mat& frame, cl_mem* buffer)
{
    cl_int res = CL_SUCCESS;
    cl_mem mem;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;

    mem = clCreateBuffer(m_context, flags, frame.total(), frame.ptr(), &res);
    if (0 == mem || CL_SUCCESS != res)
        return -1;

    cl_event event = clCreateUserEvent(m_context, &res);
    if (0 == event || CL_SUCCESS != res)
        return -1;

    res = clSetKernelArg(m_kernel, 0, sizeof(buffer), &mem);
    if (CL_SUCCESS != res)
        return -1;

    res = clSetKernelArg(m_kernel, 1, sizeof(int), &frame.step[0]);
    if (CL_SUCCESS != res)
        return -1;

    res = clSetKernelArg(m_kernel, 2, sizeof(int), &frame.rows);
    if (CL_SUCCESS != res)
        return -1;

    int cl = frame.cols / 2;
    res = clSetKernelArg(m_kernel, 3, sizeof(int), &cl);
    if (CL_SUCCESS != res)
        return -1;

    size_t size[] = { frame.cols / 2, frame.rows };
    res = clEnqueueNDRangeKernel(m_queue, m_kernel, 2, 0, size, 0, 0, 0, &event);
    if (CL_SUCCESS != res)
        return -1;

    res = clWaitForEvents(1, &event);
    if (CL_SUCCESS != res)
        return - 1;

    buffer[0] = mem;

    return  0;
}


int App::process_cl_buffer_with_opencv(cl_mem buffer, size_t step, int rows, int cols, int type, cv::UMat& u)
{
    cv::ocl::convertFromBuffer(buffer, step, rows, cols, type, u);
    cv::blur(u, u, cv::Size(7, 7), cv::Point(-3, -3));

    return 0;
}


int App::run()
{
    if(0 != initVideoSource())
        return -1;

    if(0 != initOpenCL())
        return -1;

    Mat  img_to_show;

    running = true;

    // Iterate over all frames
    while (running && nextFrame(m_frame))
    {
        cv::cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

        UMat uframe(m_frameGray.size(), m_frameGray.type());

        // work
        timerStart();

        if (process)
        {
            cl_mem buffer = 0;
            process_frame_with_open_cl(m_frameGray, &buffer);
            process_cl_buffer_with_opencv(
                buffer, m_frameGray.step[0], m_frameGray.rows, m_frameGray.cols, m_frameGray.type(), uframe);
        }
        else
        {
            m_frameGray.copyTo(uframe);
        }

        timerEnd();

        uframe.copyTo(img_to_show);

        putText(img_to_show, "OpenCL platform: " + m_platformInfo.Name(), Point(5, 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "device: " + m_deviceInfo.Name(), Point(5, 60), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "FPS: " + fpsStr(), Point(5, 90), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        imshow("opencl_interop", img_to_show);

        handleKey((char)waitKey(3));
    }

    return 0;
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
        process = !process;
        break;
    }
}


inline void App::timerStart()
{
    m_t0 = getTickCount();
}


inline void App::timerEnd()
{
    m_t1 = getTickCount();
    int64 delta = m_t1 - m_t0;
    double freq = getTickFrequency();
    m_fps = freq / delta;
}


inline string App::fpsStr() const
{
    stringstream ss;
    ss << std::fixed << std::setprecision(1) << m_fps;
    return ss.str();
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

    App app(cmd);

    try
    {
        app.run();
    }

    catch (const cv::Exception& e)
    {
        cout << "error: " << e.what() << endl;
        return 1;
    }

    catch (const std::exception& e)
    {
        cout << "error: " << e.what() << endl;
        return 1;
    }

    catch (...)
    {
        cout << "unknown exception" << endl;
        return 1;
    }

    return EXIT_SUCCESS;
} // main()

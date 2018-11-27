/*
// The example of interoperability between OpenCL and OpenCV.
// This will loop through frames of video either from input media file
// or camera device and do processing of these data in OpenCL and then
// in OpenCV. In OpenCL it does inversion of pixels in left half of frame and
// in OpenCV it does bluring in the right half of frame.
*/
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // eliminate build warning

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

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
#if defined(CL_VERSION_1_1)
        query_param(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, m_preferred_vector_width_half);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, m_native_vector_width_char);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, m_native_vector_width_short);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, m_native_vector_width_int);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, m_native_vector_width_long);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, m_native_vector_width_float);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, m_native_vector_width_double);
        query_param(id, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, m_native_vector_width_half);
#endif
        query_param(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, m_max_clock_frequency);
        query_param(id, CL_DEVICE_ADDRESS_BITS, m_address_bits);
        query_param(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, m_max_mem_alloc_size);
        query_param(id, CL_DEVICE_IMAGE_SUPPORT, m_image_support);
        query_param(id, CL_DEVICE_MAX_READ_IMAGE_ARGS, m_max_read_image_args);
        query_param(id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, m_max_write_image_args);
#if defined(CL_VERSION_2_0)
        query_param(id, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, m_max_read_write_image_args);
#endif
        query_param(id, CL_DEVICE_IMAGE2D_MAX_WIDTH, m_image2d_max_width);
        query_param(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, m_image2d_max_height);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_WIDTH, m_image3d_max_width);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, m_image3d_max_height);
        query_param(id, CL_DEVICE_IMAGE3D_MAX_DEPTH, m_image3d_max_depth);
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, m_image_max_buffer_size);
        query_param(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, m_image_max_array_size);
#endif
        query_param(id, CL_DEVICE_MAX_SAMPLERS, m_max_samplers);
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, m_image_pitch_alignment);
        query_param(id, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, m_image_base_address_alignment);
#endif
#if defined(CL_VERSION_2_0)
        query_param(id, CL_DEVICE_MAX_PIPE_ARGS, m_max_pipe_args);
        query_param(id, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, m_pipe_max_active_reservations);
        query_param(id, CL_DEVICE_PIPE_MAX_PACKET_SIZE, m_pipe_max_packet_size);
#endif
        query_param(id, CL_DEVICE_MAX_PARAMETER_SIZE, m_max_parameter_size);
        query_param(id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, m_mem_base_addr_align);
        query_param(id, CL_DEVICE_SINGLE_FP_CONFIG, m_single_fp_config);
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_DOUBLE_FP_CONFIG, m_double_fp_config);
#endif
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, m_global_mem_cache_type);
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, m_global_mem_cacheline_size);
        query_param(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, m_global_mem_cache_size);
        query_param(id, CL_DEVICE_GLOBAL_MEM_SIZE, m_global_mem_size);
        query_param(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, m_max_constant_buffer_size);
        query_param(id, CL_DEVICE_MAX_CONSTANT_ARGS, m_max_constant_args);
#if defined(CL_VERSION_2_0)
        query_param(id, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, m_max_global_variable_size);
        query_param(id, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, m_global_variable_preferred_total_size);
#endif
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
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_LINKER_AVAILABLE, m_linker_available);
#endif
        query_param(id, CL_DEVICE_EXECUTION_CAPABILITIES, m_execution_capabilities);
        query_param(id, CL_DEVICE_QUEUE_PROPERTIES, m_queue_properties);
#if defined(CL_VERSION_2_0)
        query_param(id, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, m_queue_on_host_properties);
        query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, m_queue_on_device_properties);
        query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, m_queue_on_device_preferred_size);
        query_param(id, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, m_queue_on_device_max_size);
        query_param(id, CL_DEVICE_MAX_ON_DEVICE_QUEUES, m_max_on_device_queues);
        query_param(id, CL_DEVICE_MAX_ON_DEVICE_EVENTS, m_max_on_device_events);
#endif
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_BUILT_IN_KERNELS, m_built_in_kernels);
#endif
        query_param(id, CL_DEVICE_PLATFORM, m_platform);
        query_param(id, CL_DEVICE_NAME, m_name);
        query_param(id, CL_DEVICE_VENDOR, m_vendor);
        query_param(id, CL_DRIVER_VERSION, m_driver_version);
        query_param(id, CL_DEVICE_PROFILE, m_profile);
        query_param(id, CL_DEVICE_VERSION, m_version);
#if defined(CL_VERSION_1_1)
        query_param(id, CL_DEVICE_OPENCL_C_VERSION, m_opencl_c_version);
#endif
        query_param(id, CL_DEVICE_EXTENSIONS, m_extensions);
#if defined(CL_VERSION_1_2)
        query_param(id, CL_DEVICE_PRINTF_BUFFER_SIZE, m_printf_buffer_size);
        query_param(id, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, m_preferred_interop_user_sync);
        query_param(id, CL_DEVICE_PARENT_DEVICE, m_parent_device);
        query_param(id, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, m_partition_max_sub_devices);
        query_param(id, CL_DEVICE_PARTITION_PROPERTIES, m_partition_properties);
        query_param(id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, m_partition_affinity_domain);
        query_param(id, CL_DEVICE_PARTITION_TYPE, m_partition_type);
        query_param(id, CL_DEVICE_REFERENCE_COUNT, m_reference_count);
#endif
        return CL_SUCCESS;
    }

    std::string Name() { return m_name; }

private:
    template<typename T>
    cl_int query_param(cl_device_id id, cl_device_info param, T& value)
    {
        cl_int res;
        size_t size = 0;

        res = clGetDeviceInfo(id, param, 0, 0, &size);
        if (CL_SUCCESS != res && size != 0)
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
#if defined(CL_VERSION_1_1)
    cl_uint                                   m_preferred_vector_width_half;
    cl_uint                                   m_native_vector_width_char;
    cl_uint                                   m_native_vector_width_short;
    cl_uint                                   m_native_vector_width_int;
    cl_uint                                   m_native_vector_width_long;
    cl_uint                                   m_native_vector_width_float;
    cl_uint                                   m_native_vector_width_double;
    cl_uint                                   m_native_vector_width_half;
#endif
    cl_uint                                   m_max_clock_frequency;
    cl_uint                                   m_address_bits;
    cl_ulong                                  m_max_mem_alloc_size;
    cl_bool                                   m_image_support;
    cl_uint                                   m_max_read_image_args;
    cl_uint                                   m_max_write_image_args;
#if defined(CL_VERSION_2_0)
    cl_uint                                   m_max_read_write_image_args;
#endif
    size_t                                    m_image2d_max_width;
    size_t                                    m_image2d_max_height;
    size_t                                    m_image3d_max_width;
    size_t                                    m_image3d_max_height;
    size_t                                    m_image3d_max_depth;
#if defined(CL_VERSION_1_2)
    size_t                                    m_image_max_buffer_size;
    size_t                                    m_image_max_array_size;
#endif
    cl_uint                                   m_max_samplers;
#if defined(CL_VERSION_1_2)
    cl_uint                                   m_image_pitch_alignment;
    cl_uint                                   m_image_base_address_alignment;
#endif
#if defined(CL_VERSION_2_0)
    cl_uint                                   m_max_pipe_args;
    cl_uint                                   m_pipe_max_active_reservations;
    cl_uint                                   m_pipe_max_packet_size;
#endif
    size_t                                    m_max_parameter_size;
    cl_uint                                   m_mem_base_addr_align;
    cl_device_fp_config                       m_single_fp_config;
#if defined(CL_VERSION_1_2)
    cl_device_fp_config                       m_double_fp_config;
#endif
    cl_device_mem_cache_type                  m_global_mem_cache_type;
    cl_uint                                   m_global_mem_cacheline_size;
    cl_ulong                                  m_global_mem_cache_size;
    cl_ulong                                  m_global_mem_size;
    cl_ulong                                  m_max_constant_buffer_size;
    cl_uint                                   m_max_constant_args;
#if defined(CL_VERSION_2_0)
    size_t                                    m_max_global_variable_size;
    size_t                                    m_global_variable_preferred_total_size;
#endif
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
#if defined(CL_VERSION_1_2)
    cl_bool                                   m_linker_available;
#endif
    cl_device_exec_capabilities               m_execution_capabilities;
    cl_command_queue_properties               m_queue_properties;
#if defined(CL_VERSION_2_0)
    cl_command_queue_properties               m_queue_on_host_properties;
    cl_command_queue_properties               m_queue_on_device_properties;
    cl_uint                                   m_queue_on_device_preferred_size;
    cl_uint                                   m_queue_on_device_max_size;
    cl_uint                                   m_max_on_device_queues;
    cl_uint                                   m_max_on_device_events;
#endif
#if defined(CL_VERSION_1_2)
    std::string                               m_built_in_kernels;
#endif
    cl_platform_id                            m_platform;
    std::string                               m_name;
    std::string                               m_vendor;
    std::string                               m_driver_version;
    std::string                               m_profile;
    std::string                               m_version;
#if defined(CL_VERSION_1_1)
    std::string                               m_opencl_c_version;
#endif
    std::string                               m_extensions;
#if defined(CL_VERSION_1_2)
    size_t                                    m_printf_buffer_size;
    cl_bool                                   m_preferred_interop_user_sync;
    cl_device_id                              m_parent_device;
    cl_uint                                   m_partition_max_sub_devices;
    std::vector<cl_device_partition_property> m_partition_properties;
    cl_device_affinity_domain                 m_partition_affinity_domain;
    std::vector<cl_device_partition_property> m_partition_type;
    cl_uint                                   m_reference_count;
#endif
};

} // namespace opencl


class App
{
public:
    App(CommandLineParser& cmd);
    ~App();

    int initOpenCL();
    int initVideoSource();

    int process_frame_with_open_cl(cv::Mat& frame, bool use_buffer, cl_mem* cl_buffer);
    int process_cl_buffer_with_opencv(cl_mem buffer, size_t step, int rows, int cols, int type, cv::UMat& u);
    int process_cl_image_with_opencv(cl_mem image, cv::UMat& u);

    int run();

    bool isRunning() { return m_running; }
    bool doProcess() { return m_process; }
    bool useBuffer() { return m_use_buffer; }

    void setRunning(bool running)      { m_running = running; }
    void setDoProcess(bool process)    { m_process = process; }
    void setUseBuffer(bool use_buffer) { m_use_buffer = use_buffer; }

protected:
    bool nextFrame(cv::Mat& frame) { return m_cap.read(frame); }
    void handleKey(char key);
    void timerStart();
    void timerEnd();
    std::string timeStr() const;
    std::string message() const;

private:
    bool                        m_running;
    bool                        m_process;
    bool                        m_use_buffer;

    int64                       m_t0;
    int64                       m_t1;
    float                       m_time;
    float                       m_frequency;

    string                      m_file_name;
    int                         m_camera_id;
    cv::VideoCapture            m_cap;
    cv::Mat                     m_frame;
    cv::Mat                     m_frameGray;

    opencl::PlatformInfo        m_platformInfo;
    opencl::DeviceInfo          m_deviceInfo;
    std::vector<cl_platform_id> m_platform_ids;
    cl_context                  m_context;
    cl_device_id                m_device_id;
    cl_command_queue            m_queue;
    cl_program                  m_program;
    cl_kernel                   m_kernelBuf;
    cl_kernel                   m_kernelImg;
    cl_mem                      m_img_src; // used as src in case processing of cl image
    cl_mem                      m_mem_obj;
    cl_event                    m_event;
};


App::App(CommandLineParser& cmd)
{
    cout << "\nPress ESC to exit\n" << endl;
    cout << "\n      'p' to toggle ON/OFF processing\n" << endl;
    cout << "\n       SPACE to switch between OpenCL buffer/image\n" << endl;

    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<string>("video");

    m_running    = false;
    m_process    = false;
    m_use_buffer = false;

    m_t0         = 0;
    m_t1         = 0;
    m_time       = 0.0;
    m_frequency  = (float)cv::getTickFrequency();

    m_context    = 0;
    m_device_id  = 0;
    m_queue      = 0;
    m_program    = 0;
    m_kernelBuf  = 0;
    m_kernelImg  = 0;
    m_img_src    = 0;
    m_mem_obj    = 0;
    m_event      = 0;
} // ctor


App::~App()
{
    if (m_queue)
    {
        clFinish(m_queue);
        clReleaseCommandQueue(m_queue);
        m_queue = 0;
    }

    if (m_program)
    {
        clReleaseProgram(m_program);
        m_program = 0;
    }

    if (m_img_src)
    {
        clReleaseMemObject(m_img_src);
        m_img_src = 0;
    }

    if (m_mem_obj)
    {
        clReleaseMemObject(m_mem_obj);
        m_mem_obj = 0;
    }

    if (m_event)
    {
        clReleaseEvent(m_event);
    }

    if (m_kernelBuf)
    {
        clReleaseKernel(m_kernelBuf);
        m_kernelBuf = 0;
    }

    if (m_kernelImg)
    {
        clReleaseKernel(m_kernelImg);
        m_kernelImg = 0;
    }

    if (m_device_id)
    {
        clReleaseDevice(m_device_id);
        m_device_id = 0;
    }

    if (m_context)
    {
        clReleaseContext(m_context);
        m_context = 0;
    }
} // dtor


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
            "void bitwise_inv_buf_8uC1("
            "    __global unsigned char* pSrcDst,"
            "             int            srcDstStep,"
            "             int            rows,"
            "             int            cols)"
            "{"
            "    int x = get_global_id(0);"
            "    int y = get_global_id(1);"
            "    int idx = mad24(y, srcDstStep, x);"
            "    pSrcDst[idx] = ~pSrcDst[idx];"
            "}"
            "__kernel "
            "void bitwise_inv_img_8uC1("
            "    read_only  image2d_t srcImg,"
            "    write_only image2d_t dstImg)"
            "{"
            "    int x = get_global_id(0);"
            "    int y = get_global_id(1);"
            "    int2 coord = (int2)(x, y);"
            "    uint4 val = read_imageui(srcImg, coord);"
            "    val.x = (~val.x) & 0x000000FF;"
            "    write_imageui(dstImg, coord, val);"
            "}";
        size_t len = strlen(kernelSrc);
        m_program = clCreateProgramWithSource(m_context, 1, &kernelSrc, &len, &res);
        if (0 == m_program || CL_SUCCESS != res)
            return -1;

        res = clBuildProgram(m_program, 1, &m_device_id, 0, 0, 0);
        if (CL_SUCCESS != res)
            return -1;

        m_kernelBuf = clCreateKernel(m_program, "bitwise_inv_buf_8uC1", &res);
        if (0 == m_kernelBuf || CL_SUCCESS != res)
            return -1;

        m_kernelImg = clCreateKernel(m_program, "bitwise_inv_img_8uC1", &res);
        if (0 == m_kernelImg || CL_SUCCESS != res)
            return -1;

        m_platformInfo.QueryInfo(m_platform_ids[i]);
        m_deviceInfo.QueryInfo(m_device_id);

        // attach OpenCL context to OpenCV
        cv::ocl::attachContext(m_platformInfo.Name(), m_platform_ids[i], m_context, m_device_id);

        break;
    }

    return m_context != 0 ? CL_SUCCESS : -1;
} // initOpenCL()


int App::initVideoSource()
{
    try
    {
        if (!m_file_name.empty() && m_camera_id == -1)
        {
            m_cap.open(m_file_name.c_str());
            if (!m_cap.isOpened())
                throw std::runtime_error(std::string("can't open video file: " + m_file_name));
        }
        else if (m_camera_id != -1)
        {
            m_cap.open(m_camera_id);
            if (!m_cap.isOpened())
            {
                std::stringstream msg;
                msg << "can't open camera: " << m_camera_id;
                throw std::runtime_error(msg.str());
            }
        }
        else
            throw std::runtime_error(std::string("specify video source"));
    }

    catch (const std::exception e)
    {
        cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} // initVideoSource()


// this function is an example of "typical" OpenCL processing pipeline
// It creates OpenCL buffer or image, depending on use_buffer flag,
// from input media frame and process these data
// (inverts each pixel value in half of frame) with OpenCL kernel
int App::process_frame_with_open_cl(cv::Mat& frame, bool use_buffer, cl_mem* mem_obj)
{
    cl_int res = CL_SUCCESS;

    CV_Assert(mem_obj);

    cl_kernel kernel = 0;
    cl_mem mem = mem_obj[0];

    if (0 == mem || 0 == m_img_src)
    {
        // allocate/delete cl memory objects every frame for the simplicity.
        // in real applicaton more efficient pipeline can be built.

        if (use_buffer)
        {
            cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;

            mem = clCreateBuffer(m_context, flags, frame.total(), frame.ptr(), &res);
            if (0 == mem || CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 0, sizeof(cl_mem), &mem);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 1, sizeof(int), &frame.step[0]);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 2, sizeof(int), &frame.rows);
            if (CL_SUCCESS != res)
                return -1;

            int cols2 = frame.cols / 2;
            res = clSetKernelArg(m_kernelBuf, 3, sizeof(int), &cols2);
            if (CL_SUCCESS != res)
                return -1;

            kernel = m_kernelBuf;
        }
        else
        {
            cl_mem_flags flags_src = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;

            cl_image_format fmt;
            fmt.image_channel_order     = CL_R;
            fmt.image_channel_data_type = CL_UNSIGNED_INT8;

            cl_image_desc desc_src;
            desc_src.image_type        = CL_MEM_OBJECT_IMAGE2D;
            desc_src.image_width       = frame.cols;
            desc_src.image_height      = frame.rows;
            desc_src.image_depth       = 0;
            desc_src.image_array_size  = 0;
            desc_src.image_row_pitch   = frame.step[0];
            desc_src.image_slice_pitch = 0;
            desc_src.num_mip_levels    = 0;
            desc_src.num_samples       = 0;
            desc_src.buffer            = 0;
            m_img_src = clCreateImage(m_context, flags_src, &fmt, &desc_src, frame.ptr(), &res);
            if (0 == m_img_src || CL_SUCCESS != res)
                return -1;

            cl_mem_flags flags_dst = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;

            cl_image_desc desc_dst;
            desc_dst.image_type        = CL_MEM_OBJECT_IMAGE2D;
            desc_dst.image_width       = frame.cols;
            desc_dst.image_height      = frame.rows;
            desc_dst.image_depth       = 0;
            desc_dst.image_array_size  = 0;
            desc_dst.image_row_pitch   = 0;
            desc_dst.image_slice_pitch = 0;
            desc_dst.num_mip_levels    = 0;
            desc_dst.num_samples       = 0;
            desc_dst.buffer            = 0;
            mem = clCreateImage(m_context, flags_dst, &fmt, &desc_dst, 0, &res);
            if (0 == mem || CL_SUCCESS != res)
                return -1;

            size_t origin[] = { 0, 0, 0 };
            size_t region[] = { (size_t)frame.cols, (size_t)frame.rows, 1 };
            res = clEnqueueCopyImage(m_queue, m_img_src, mem, origin, origin, region, 0, 0, &m_event);
            if (CL_SUCCESS != res)
                return -1;

            res = clWaitForEvents(1, &m_event);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelImg, 0, sizeof(cl_mem), &m_img_src);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelImg, 1, sizeof(cl_mem), &mem);
            if (CL_SUCCESS != res)
                return -1;

            kernel = m_kernelImg;
        }
    }

    m_event = clCreateUserEvent(m_context, &res);
    if (0 == m_event || CL_SUCCESS != res)
        return -1;

    // process left half of frame in OpenCL
    size_t size[] = { (size_t)frame.cols / 2, (size_t)frame.rows };
    res = clEnqueueNDRangeKernel(m_queue, kernel, 2, 0, size, 0, 0, 0, &m_event);
    if (CL_SUCCESS != res)
        return -1;

    res = clWaitForEvents(1, &m_event);
    if (CL_SUCCESS != res)
        return - 1;

    mem_obj[0] = mem;

    return  0;
}


// this function is an example of interoperability between OpenCL buffer
// and OpenCV UMat objects. It converts (without copying data) OpenCL buffer
// to OpenCV UMat and then do blur on these data
int App::process_cl_buffer_with_opencv(cl_mem buffer, size_t step, int rows, int cols, int type, cv::UMat& u)
{
    cv::ocl::convertFromBuffer(buffer, step, rows, cols, type, u);

    // process right half of frame in OpenCV
    cv::Point pt(u.cols / 2, 0);
    cv::Size  sz(u.cols / 2, u.rows);
    cv::Rect roi(pt, sz);
    cv::UMat uroi(u, roi);
    cv::blur(uroi, uroi, cv::Size(7, 7), cv::Point(-3, -3));

    if (buffer)
        clReleaseMemObject(buffer);
    m_mem_obj = 0;

    return 0;
}


// this function is an example of interoperability between OpenCL image
// and OpenCV UMat objects. It converts OpenCL image
// to OpenCV UMat and then do blur on these data
int App::process_cl_image_with_opencv(cl_mem image, cv::UMat& u)
{
    cv::ocl::convertFromImage(image, u);

    // process right half of frame in OpenCV
    cv::Point pt(u.cols / 2, 0);
    cv::Size  sz(u.cols / 2, u.rows);
    cv::Rect roi(pt, sz);
    cv::UMat uroi(u, roi);
    cv::blur(uroi, uroi, cv::Size(7, 7), cv::Point(-3, -3));

    if (image)
        clReleaseMemObject(image);
    m_mem_obj = 0;

    if (m_img_src)
        clReleaseMemObject(m_img_src);
    m_img_src = 0;

    return 0;
}


int App::run()
{
    if (0 != initOpenCL())
        return -1;

    if (0 != initVideoSource())
        return -1;

    Mat img_to_show;

    // set running state until ESC pressed
    setRunning(true);
    // set process flag to show some data processing
    // can be toggled on/off by 'p' button
    setDoProcess(true);
    // set use buffer flag,
    // when it is set to true, will demo interop opencl buffer and cv::Umat,
    // otherwise demo interop opencl image and cv::UMat
    // can be switched on/of by SPACE button
    setUseBuffer(true);

    // Iterate over all frames
    while (isRunning() && nextFrame(m_frame))
    {
        cv::cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

        UMat uframe;

        // work
        timerStart();

        if (doProcess())
        {
            process_frame_with_open_cl(m_frameGray, useBuffer(), &m_mem_obj);

            if (useBuffer())
                process_cl_buffer_with_opencv(
                    m_mem_obj, m_frameGray.step[0], m_frameGray.rows, m_frameGray.cols, m_frameGray.type(), uframe);
            else
                process_cl_image_with_opencv(m_mem_obj, uframe);
        }
        else
        {
            m_frameGray.copyTo(uframe);
        }

        timerEnd();

        uframe.copyTo(img_to_show);

        putText(img_to_show, "Version : " + m_platformInfo.Version(), Point(5, 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Name : " + m_platformInfo.Name(), Point(5, 60), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Device : " + m_deviceInfo.Name(), Point(5, 90), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        cv::String memtype = useBuffer() ? "buffer" : "image";
        putText(img_to_show, "interop with OpenCL " + memtype, Point(5, 120), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Time : " + timeStr() + " msec", Point(5, 150), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);

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
        setRunning(false);
        break;

    case ' ':
        setUseBuffer(!useBuffer());
        break;

    case 'p':
    case 'P':
        setDoProcess( !doProcess() );
        break;

    default:
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
    m_time = (delta / m_frequency) * 1000; // units msec
}


inline string App::timeStr() const
{
    stringstream ss;
    ss << std::fixed << std::setprecision(1) << m_time;
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

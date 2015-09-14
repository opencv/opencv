/* origin: libva-1.3.1/test/decode/mpeg2vldemo.cpp */

/*
 * Copyright (c) 2007-2008 Intel Corporation. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL PRECISION INSIGHT AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <string>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <va/va.h>

#include "display.cpp.inc"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/va_intel.hpp"
#include "cvconfig.h"

#define CHECK_VASTATUS(va_status,func)                                  \
if (va_status != VA_STATUS_SUCCESS) {                                   \
    fprintf(stderr,"%s:%s (%d) failed(status=0x%08x),exit\n", __func__, func, __LINE__, va_status); \
    exit(1);                                                            \
}

/* Data dump of a 16x16 MPEG2 video clip,it has one I frame
 */
static unsigned char mpeg2_clip[]={
    0x00,0x00,0x01,0xb3,0x01,0x00,0x10,0x13,0xff,0xff,0xe0,0x18,0x00,0x00,0x01,0xb5,
    0x14,0x8a,0x00,0x01,0x00,0x00,0x00,0x00,0x01,0xb8,0x00,0x08,0x00,0x00,0x00,0x00,
    0x01,0x00,0x00,0x0f,0xff,0xf8,0x00,0x00,0x01,0xb5,0x8f,0xff,0xf3,0x41,0x80,0x00,
    0x00,0x01,0x01,0x13,0xe1,0x00,0x15,0x81,0x54,0xe0,0x2a,0x05,0x43,0x00,0x2d,0x60,
    0x18,0x01,0x4e,0x82,0xb9,0x58,0xb1,0x83,0x49,0xa4,0xa0,0x2e,0x05,0x80,0x4b,0x7a,
    0x00,0x01,0x38,0x20,0x80,0xe8,0x05,0xff,0x60,0x18,0xe0,0x1d,0x80,0x98,0x01,0xf8,
    0x06,0x00,0x54,0x02,0xc0,0x18,0x14,0x03,0xb2,0x92,0x80,0xc0,0x18,0x94,0x42,0x2c,
    0xb2,0x11,0x64,0xa0,0x12,0x5e,0x78,0x03,0x3c,0x01,0x80,0x0e,0x80,0x18,0x80,0x6b,
    0xca,0x4e,0x01,0x0f,0xe4,0x32,0xc9,0xbf,0x01,0x42,0x69,0x43,0x50,0x4b,0x01,0xc9,
    0x45,0x80,0x50,0x01,0x38,0x65,0xe8,0x01,0x03,0xf3,0xc0,0x76,0x00,0xe0,0x03,0x20,
    0x28,0x18,0x01,0xa9,0x34,0x04,0xc5,0xe0,0x0b,0x0b,0x04,0x20,0x06,0xc0,0x89,0xff,
    0x60,0x12,0x12,0x8a,0x2c,0x34,0x11,0xff,0xf6,0xe2,0x40,0xc0,0x30,0x1b,0x7a,0x01,
    0xa9,0x0d,0x00,0xac,0x64
};

/* hardcoded here without a bitstream parser helper
 * please see picture mpeg2-I.jpg for bitstream details
 */
static VAPictureParameterBufferMPEG2 pic_param={
  horizontal_size:16,
  vertical_size:16,
  forward_reference_picture:0xffffffff,
  backward_reference_picture:0xffffffff,
  picture_coding_type:1,
  f_code:0xffff,
  {
      {
        intra_dc_precision:0,
        picture_structure:3,
        top_field_first:0,
        frame_pred_frame_dct:1,
        concealment_motion_vectors:0,
        q_scale_type:0,
        intra_vlc_format:0,
        alternate_scan:0,
        repeat_first_field:0,
        progressive_frame:1 ,
        is_first_field:1
      },
  }
};

/* see MPEG2 spec65 for the defines of matrix */
static VAIQMatrixBufferMPEG2 iq_matrix = {
  load_intra_quantiser_matrix:1,
  load_non_intra_quantiser_matrix:1,
  load_chroma_intra_quantiser_matrix:0,
  load_chroma_non_intra_quantiser_matrix:0,
  intra_quantiser_matrix:{
         8, 16, 16, 19, 16, 19, 22, 22,
        22, 22, 22, 22, 26, 24, 26, 27,
        27, 27, 26, 26, 26, 26, 27, 27,
        27, 29, 29, 29, 34, 34, 34, 29,
        29, 29, 27, 27, 29, 29, 32, 32,
        34, 34, 37, 38, 37, 35, 35, 34,
        35, 38, 38, 40, 40, 40, 48, 48,
        46, 46, 56, 56, 58, 69, 69, 83
    },
  non_intra_quantiser_matrix:{16},
  chroma_intra_quantiser_matrix:{0},
  chroma_non_intra_quantiser_matrix:{0}
};

#if 1
static VASliceParameterBufferMPEG2 slice_param={
  slice_data_size:150,
  slice_data_offset:0,
  slice_data_flag:0,
  macroblock_offset:38, /* 4byte + 6bits=38bits */
  slice_horizontal_position:0,
  slice_vertical_position:0,
  quantiser_scale_code:2,
  intra_slice_flag:0
};
#endif

#define CLIP_WIDTH  16
#define CLIP_HEIGHT 16

class Timer
{
public:
    enum UNITS
    {
        USEC = 0,
        MSEC,
        SEC
    };

    Timer() : m_t0(0), m_diff(0)
    {
        m_tick_frequency = (float)cv::getTickFrequency();

        m_unit_mul[USEC] = 1000000;
        m_unit_mul[MSEC] = 1000;
        m_unit_mul[SEC]  = 1;
    }

    void clear()
    {
        m_t0 = m_diff = 0;
    }

    void start()
    {
        m_t0 = cv::getTickCount();
    }

    void stop()
    {
        m_diff = cv::getTickCount() - m_t0;
    }

    float time(UNITS u = MSEC)
    {
        float sec = m_diff / m_tick_frequency;

        return sec * m_unit_mul[u];
    }

public:
    float m_tick_frequency;
    int64 m_t0;
    int64 m_diff;
    int   m_unit_mul[3];
};

static void dumpSurface(VADisplay display, VASurfaceID surface_id, const char* fileName, bool doInterop)
{
    VAStatus va_status;

    va_status = vaSyncSurface(display, surface_id);
    CHECK_VASTATUS(va_status, "vaSyncSurface");

    VAImage image;
    va_status = vaDeriveImage(display, surface_id, &image);
    CHECK_VASTATUS(va_status, "vaDeriveImage");

    unsigned char* buffer = 0;
    va_status = vaMapBuffer(display, image.buf, (void **)&buffer);
    CHECK_VASTATUS(va_status, "vaMapBuffer");

    CV_Assert(image.format.fourcc == VA_FOURCC_NV12);
/*
    printf("image.format.fourcc = 0x%08x\n", image.format.fourcc);
    printf("image.[width x height] = %d x %d\n", image.width, image.height);
    printf("image.data_size = %d\n", image.data_size);
    printf("image.num_planes = %d\n", image.num_planes);
    printf("image.pitches[0..2] = 0x%08x 0x%08x 0x%08x\n", image.pitches[0], image.pitches[1], image.pitches[2]);
    printf("image.offsets[0..2] = 0x%08x 0x%08x 0x%08x\n", image.offsets[0], image.offsets[1], image.offsets[2]);
*/
    std::string fn = std::string(fileName) + std::string(doInterop ? ".on" : ".off");
    FILE* out = fopen(fn.c_str(), "wb");
    if (!out)
    {
        perror(fileName);
        exit(1);
    }
    fwrite(buffer, 1, image.data_size, out);
    fclose(out);

    vaUnmapBuffer(display, image.buf);
    CHECK_VASTATUS(va_status, "vaUnmapBuffer");

    vaDestroyImage(display, image.image_id);
    CHECK_VASTATUS(va_status, "vaDestroyImage");
}

static float run(const char* fn1, const char* fn2, bool doInterop)
{
    VAEntrypoint entrypoints[5];
    int num_entrypoints,vld_entrypoint;
    VAConfigAttrib attrib;
    VAConfigID config_id;
    VASurfaceID surface_id;
    VAContextID context_id;
    VABufferID pic_param_buf,iqmatrix_buf,slice_param_buf,slice_data_buf;
    VAStatus va_status;
    Timer t;

    cv::va_intel::ocl::initializeContextFromVA(va::display, doInterop);

    va_status = vaQueryConfigEntrypoints(va::display, VAProfileMPEG2Main, entrypoints,
                                         &num_entrypoints);
    CHECK_VASTATUS(va_status, "vaQueryConfigEntrypoints");

    for (vld_entrypoint = 0; vld_entrypoint < num_entrypoints; vld_entrypoint++) {
        if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
            break;
    }
    if (vld_entrypoint == num_entrypoints) {
        /* not find VLD entry point */
        assert(0);
    }

    /* Assuming finding VLD, find out the format for the render target */
    attrib.type = VAConfigAttribRTFormat;
    vaGetConfigAttributes(va::display, VAProfileMPEG2Main, VAEntrypointVLD,
                          &attrib, 1);
    if ((attrib.value & VA_RT_FORMAT_YUV420) == 0) {
        /* not find desired YUV420 RT format */
        assert(0);
    }

    va_status = vaCreateConfig(va::display, VAProfileMPEG2Main, VAEntrypointVLD,
                               &attrib, 1,&config_id);
    CHECK_VASTATUS(va_status, "vaCreateConfig");

    va_status = vaCreateSurfaces(
        va::display,
        VA_RT_FORMAT_YUV420, CLIP_WIDTH, CLIP_HEIGHT,
        &surface_id, 1,
        NULL, 0
    );
    CHECK_VASTATUS(va_status, "vaCreateSurfaces");

    /* Create a context for this decode pipe */
    va_status = vaCreateContext(va::display, config_id,
                                CLIP_WIDTH,
                                ((CLIP_HEIGHT+15)/16)*16,
                                VA_PROGRESSIVE,
                                &surface_id,
                                1,
                                &context_id);
    CHECK_VASTATUS(va_status, "vaCreateContext");

    va_status = vaCreateBuffer(va::display, context_id,
                               VAPictureParameterBufferType,
                               sizeof(VAPictureParameterBufferMPEG2),
                               1, &pic_param,
                               &pic_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer");

    va_status = vaCreateBuffer(va::display, context_id,
                               VAIQMatrixBufferType,
                               sizeof(VAIQMatrixBufferMPEG2),
                               1, &iq_matrix,
                               &iqmatrix_buf );
    CHECK_VASTATUS(va_status, "vaCreateBuffer");

    va_status = vaCreateBuffer(va::display, context_id,
                               VASliceParameterBufferType,
                               sizeof(VASliceParameterBufferMPEG2),
                               1,
                               &slice_param, &slice_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer");

    va_status = vaCreateBuffer(va::display, context_id,
                               VASliceDataBufferType,
                               0xc4-0x2f+1,
                               1,
                               mpeg2_clip+0x2f,
                               &slice_data_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer");

    va_status = vaBeginPicture(va::display, context_id, surface_id);
    CHECK_VASTATUS(va_status, "vaBeginPicture");

    va_status = vaRenderPicture(va::display,context_id, &pic_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");

    va_status = vaRenderPicture(va::display,context_id, &iqmatrix_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");

    va_status = vaRenderPicture(va::display,context_id, &slice_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");

    va_status = vaRenderPicture(va::display,context_id, &slice_data_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");

    va_status = vaEndPicture(va::display,context_id);
    CHECK_VASTATUS(va_status, "vaEndPicture");

    va_status = vaSyncSurface(va::display, surface_id);
    CHECK_VASTATUS(va_status, "vaSyncSurface");

    dumpSurface(va::display, surface_id, fn1, doInterop);

    cv::Size size(CLIP_WIDTH,CLIP_HEIGHT);
    cv::UMat u;

    cv::va_intel::convertFromVASurface(va::display, surface_id, size, u);
    cv::blur(u, u, cv::Size(7, 7), cv::Point(-3, -3));
    cv::va_intel::convertToVASurface(va::display, u, surface_id, size);
    t.start();
    cv::va_intel::convertFromVASurface(va::display, surface_id, size, u);
    cv::blur(u, u, cv::Size(7, 7), cv::Point(-3, -3));
    cv::va_intel::convertToVASurface(va::display, u, surface_id, size);
    t.stop();

    dumpSurface(va::display, surface_id, fn2, doInterop);

    vaDestroySurfaces(va::display,&surface_id,1);
    vaDestroyConfig(va::display,config_id);
    vaDestroyContext(va::display,context_id);

    return t.time(Timer::MSEC);
}

class CmdlineParser
{
public:
    CmdlineParser(int argc, char** argv):
        m_argc(argc), m_argv(argv)
        {}
    // true => go, false => usage/exit; extra args/unknown options are ignored for simplicity
    bool run()
        {
            int n = 0;
            m_files[0] = m_files[1] = 0;
#if defined(HAVE_VA_INTEL)
            m_interop = true;
#elif defined(HAVE_VA)
            m_interop = false;
#endif //HAVE_VA_INTEL / HAVE_VA
            for (int i = 1; i < m_argc; ++i)
            {
                const char *arg = m_argv[i];
                if (arg[0] == '-') // option
                {
#if defined(HAVE_VA_INTEL)
                    if (!strcmp(arg, "-f"))
                        m_interop = false;
#endif //HAVE_VA_INTEL
                }
                else // parameter
                {
                    if (n < 2)
                        m_files[n++] = arg;
                }
            }
            return bool(n >= 2);
        }
    bool isInterop() const
        {
            return m_interop;
        }
    const char* getFile(int n) const
        {
            return ((n >= 0) && (n < 2)) ? m_files[n] : 0;
        }
private:
    int m_argc;
    char** m_argv;
    const char* m_files[2];
    bool m_interop;
};

int main(int argc, char** argv)
{
    CmdlineParser cmd(argc, argv);
    if (!cmd.run())
    {
        fprintf(stderr,
#if defined(HAVE_VA_INTEL)
                "Usage: va_intel_interop [-f] file1 file2\n\n"
                "Interop ON/OFF version\n\n"
                "where:  -f    option indicates interop is off (fallback mode); interop is on by default\n"
#elif defined(HAVE_VA)
                "Usage: va_intel_interop file1 file2\n\n"
                "Interop OFF only version\n\n"
                "where:\n"
#endif //HAVE_VA_INTEL / HAVE_VA
                "        file1 is to be created, contains original surface data (NV12)\n"
                "        file2 is to be created, contains processed surface data (NV12)\n");
        exit(0);
    }

    if (!va::openDisplay())
    {
        fprintf(stderr, "Failed to open VA display for CL-VA interoperability\n");
        exit(1);
    }
    fprintf(stderr, "VA display opened successfully\n");

    const char* file0 = cmd.getFile(0);
    const char* file1 = cmd.getFile(1);
    bool doInterop = cmd.isInterop();

    float time = run(file0, file1, doInterop);

    fprintf(stderr, "Interop %s: processing time, msec: %7.3f\n", (doInterop ? "ON " : "OFF"), time);

    va::closeDisplay();
    return 0;
}

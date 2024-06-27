#include "fastcv_hal.hpp"
#include "fastcv.h"
#include <cstdint>
#include <stdio.h>

static const char* getFastCVErrorString(fcvStatus status)
{
    switch(status)
    {
        case FASTCV_SUCCESS: return "Succesful";
        case FASTCV_EFAIL: return "General failure";
        case FASTCV_EUNALIGNPARAM: return "Unaligned pointer parameter";
        case FASTCV_EBADPARAM: return "Bad parameters";
        case FASTCV_EINVALSTATE: return "Called at invalid state";
        case FASTCV_ENORES: return "Insufficient resources, memory, thread, etc";
        case FASTCV_EUNSUPPORTED: return "Unsupported feature";
        case FASTCV_EHWQDSP: return "Hardware QDSP failed to respond";
        case FASTCV_EHWGPU: return "Hardware GPU failed to respond";
    }
}

int fastcv_hal_add_8u(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h)
{
    printf("width: %d, height: %d\n", w, h);
    printf("astep: %zu, bstep: %zu, cstep: %zu\n", astep, bstep, cstep);

    // stride shpuld be miltiple of 8
    if ((astep % 8 != 0) || (bstep % 8 != 0) || (cstep % 8 != 0))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // 128-bit alignment check
    if (((uintptr_t)a % 16) || ((uintptr_t)b % 16) || ((uintptr_t)c % 16))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (h == 1)
    {
        astep = w*sizeof(uchar);
        bstep = w*sizeof(uchar);
        cstep = w*sizeof(uchar);
    }

    fcvStatus status = fcvAddu8(a, w, h, astep, b, bstep, FASTCV_CONVERT_POLICY_SATURATE, c, cstep);
    if (status == FASTCV_SUCCESS)
        return CV_HAL_ERROR_OK;
    else
    {
        printf("FastCV error: %s\n", getFastCVErrorString(status));
        return CV_HAL_ERROR_UNKNOWN;
    }
}


int fastcv_hal_setto_mask(uchar *dst_data, int dst_step, int dst_cols, int dst_rows,
                          const uchar* mask_data, int mask_step, uchar *value_data, int value_size)
{
    // 128-bit alignment check
    if (((uintptr_t)dst_data % 16) || ((uintptr_t)mask_data % 16))
    {
        printf("ptr %% 16 break\n");
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // stride should be miltiple of 8
    // if ((dst_step % 8) || (mask_step % 8))
    // {
    //     printf("stride %% 8 break\n");
    //     return CV_HAL_ERROR_NOT_IMPLEMENTED;
    // }

    //printf("HAL setto\n");

    switch (value_size)
    {
        case 1:
        {
            uchar value = *value_data;
            fcvSetElementsu8(dst_data, dst_cols, dst_rows, dst_step, value, mask_data, mask_step);
            break;
        }
        case 3:
        {
            uchar v0, v1, v2;
            v0 = value_data[0]; v1 = value_data[1]; v2 = value_data[2];
            fcvSetElementsc3u8(dst_data, dst_cols, dst_rows, dst_step, v0, v1, v2, mask_data, mask_step);
            break;
        }
        case 4:
        {
            int32_t value = ((int32_t*)value_data)[0];
            fcvSetElementss32((int32_t*)dst_data, dst_cols, dst_rows, dst_step, value, mask_data, mask_step);
            break;
        }
        case 3*4:
        {
            int32_t v0, v1, v2;
            v0 = ((int32_t*)value_data)[0]; v1 = ((int32_t*)value_data)[1]; v2 = ((int32_t*)value_data)[2];
            fcvSetElementsc3s32((int32_t*)dst_data, dst_cols, dst_rows, dst_step, v0, v1, v2, mask_data, mask_step);
            break;
        }
        case 4*4:
        {
            int32_t v0, v1, v2, v3;
            v0 = ((int32_t*)value_data)[0]; v1 = ((int32_t*)value_data)[1]; v2 = ((int32_t*)value_data)[2]; v3 = ((int32_t*)value_data)[3];
            fcvSetElementsc4s32((int*)dst_data, dst_cols, dst_rows, dst_step, v0, v1, v2, v3, mask_data, mask_step);
            break;
        }
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

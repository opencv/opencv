#include "ndsrvp_hal.hpp"

int ndsrvp_integral(int depth, int sdepth, int sqdepth,
    const uchar* src, size_t _srcstep,
    uchar* _sum, size_t _sumstep,
    uchar* _sqsum, size_t,
    uchar* _tilted, size_t,
    int width, int height, int cn)
{
    if (!(depth == CV_8U && sdepth == CV_32S))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int* sum = (int*)_sum;
    double* sqsum = (double*)_sqsum;
    int* tilted = (int*)_tilted;

    if (sqsum || tilted || cn > 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    sqdepth = sqdepth;
    width *= cn;

    memset(sum, 0, (width + cn) * sizeof(int));

    if (cn == 1) {
        for (int i = 0; i < height; ++i) {
            const uchar* src_row = src + _srcstep * i;
            int* prev_sum_row = (int*)((uchar*)sum + _sumstep * i) + 1;
            int* sum_row = (int*)((uchar*)sum + _sumstep * (i + 1)) + 1;

            sum_row[-1] = 0;

            int32x2_t prev = { 0, 0 };
            int j = 0;

            for (; j + 4 <= width; j += 4) {
                uint8x4_t vs8x4 = *(uint8x4_t*)(src_row + j);
                int16x4_t vs16x4 = (int16x4_t)__nds__pkbb32(__nds__zunpkd832((unsigned int)vs8x4),
                    __nds__zunpkd810((unsigned int)vs8x4));

                vs16x4 += (int16x4_t)((unsigned long)vs16x4 << 16); // gcc vector extension
                vs16x4 += (int16x4_t)((unsigned long)vs16x4 << 32); // '+' is add16

                *(int32x2_t*)(sum_row + j) = (int32x2_t) { vs16x4[0], vs16x4[1] } + *(int32x2_t*)(prev_sum_row + j) + prev; // '+' is add32
                *(int32x2_t*)(sum_row + j + 2) = (int32x2_t) { vs16x4[2], vs16x4[3] } + *(int32x2_t*)(prev_sum_row + j + 2) + prev;

                prev += vs16x4[3]; // prev += (int32x2_t){vs16x4[3], vs16x4[3]};
            }

            for (int v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
        }
    } else if (cn == 2) {
        for (int i = 0; i < height; ++i) {
            const uchar* src_row = src + _srcstep * i;
            int* prev_sum_row = (int*)((uchar*)sum + _sumstep * i) + cn;
            int* sum_row = (int*)((uchar*)sum + _sumstep * (i + 1)) + cn;

            sum_row[-1] = sum_row[-2] = 0;

            int32x2_t prev = { 0, 0 };
            int j = 0;
            for (; j + 4 * cn <= width; j += 4 * cn) {
                uint8x8_t vs8x8 = *(uint8x8_t*)(src_row + j);

                uint16x4_t vs16x4_1 = __nds__v_zunpkd820(vs8x8);
                uint16x4_t vs16x4_2 = __nds__v_zunpkd831(vs8x8);

                vs16x4_1 += (int16x4_t)((unsigned long)vs16x4_1 << 16);
                vs16x4_1 += (int16x4_t)((unsigned long)vs16x4_1 << 32);

                vs16x4_2 += (int16x4_t)((unsigned long)vs16x4_2 << 16);
                vs16x4_2 += (int16x4_t)((unsigned long)vs16x4_2 << 32);

                *(int32x2_t*)(sum_row + j) = (int32x2_t) { vs16x4_1[0], vs16x4_2[0] } + *(int32x2_t*)(prev_sum_row + j) + prev;
                *(int32x2_t*)(sum_row + j + 2) = (int32x2_t) { vs16x4_1[1], vs16x4_2[1] } + *(int32x2_t*)(prev_sum_row + j + 2) + prev;
                *(int32x2_t*)(sum_row + j + 2 * 2) = (int32x2_t) { vs16x4_1[2], vs16x4_2[2] } + *(int32x2_t*)(prev_sum_row + j + 2 * 2) + prev;
                *(int32x2_t*)(sum_row + j + 2 * 3) = (int32x2_t) { vs16x4_1[3], vs16x4_2[3] } + *(int32x2_t*)(prev_sum_row + j + 2 * 3) + prev;

                prev += (int32x2_t) { vs16x4_1[3], vs16x4_2[3] };
            }

            for (int v2 = sum_row[j - 1] - prev_sum_row[j - 1],
                     v1 = sum_row[j - 2] - prev_sum_row[j - 2];
                 j < width; j += 2) {
                sum_row[j] = (v1 += src_row[j]) + prev_sum_row[j];
                sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
            }
        }
    } else if (cn == 3) {
        for (int i = 0; i < height; ++i) {
            const uchar* src_row = src + _srcstep * i;
            int* prev_sum_row = (int*)((uchar*)sum + _sumstep * i) + cn;
            int* sum_row = (int*)((uchar*)sum + _sumstep * (i + 1)) + cn;

            sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

            int32x2_t prev_ptr[2] = { { 0, 0 }, { 0, 0 } };
            int j = 0;
            for (; j + 3 <= width; j += 3) {
                int8x4_t vs8x4 = *(int8x4_t*)(src_row + j);

                // [ 0 | 2 | 1 | 3 ]
                int16x4_t vs16x4 = (int16x4_t)__nds__pkbb32(__nds__zunpkd831((unsigned int)vs8x4), __nds__zunpkd820((unsigned int)vs8x4));

                // [ b | t | b | t ]
                prev_ptr[0] += (int32x2_t)__nds__pkbb16(0, (unsigned long)vs16x4);
                prev_ptr[1] += (int32x2_t)__nds__pktt16(0, (unsigned long)vs16x4);

                *(int32x4_t*)(sum_row + j) = *(int32x4_t*)(prev_sum_row + j) + *(int32x4_t*)prev_ptr;
            }

            for (int v3 = sum_row[j - 1] - prev_sum_row[j - 1],
                     v2 = sum_row[j - 2] - prev_sum_row[j - 2],
                     v1 = sum_row[j - 3] - prev_sum_row[j - 3];
                 j < width; j += 3) {
                sum_row[j] = (v1 += src_row[j]) + prev_sum_row[j];
                sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
            }
        }
    } else if (cn == 4) {
        for (int i = 0; i < height; ++i) {
            const uchar* src_row = src + _srcstep * i;
            int* prev_sum_row = (int*)((uchar*)sum + _sumstep * i) + cn;
            int* sum_row = (int*)((uchar*)sum + _sumstep * (i + 1)) + cn;

            sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

            int32x2_t prev_ptr[2] = { { 0, 0 }, { 0, 0 } };
            int j = 0;
            for (; j + 4 <= width; j += 4) {
                int8x4_t vs8x4 = *(int8x4_t*)(src_row + j);

                // [ 0 | 2 | 1 | 3 ]
                int16x4_t vs16x4 = (int16x4_t)__nds__pkbb32(__nds__zunpkd831((unsigned int)vs8x4), __nds__zunpkd820((unsigned int)vs8x4));

                // [ b | t | b | t ]
                prev_ptr[0] += (int32x2_t)__nds__pkbb16(0, (unsigned long)vs16x4);
                prev_ptr[1] += (int32x2_t)__nds__pktt16(0, (unsigned long)vs16x4);

                *(int32x4_t*)(sum_row + j) = *(int32x4_t*)(prev_sum_row + j) + *(int32x4_t*)prev_ptr;
            }

            for (int v4 = sum_row[j - 1] - prev_sum_row[j - 1],
                     v3 = sum_row[j - 2] - prev_sum_row[j - 2],
                     v2 = sum_row[j - 3] - prev_sum_row[j - 3],
                     v1 = sum_row[j - 4] - prev_sum_row[j - 4];
                 j < width; j += 4) {
                sum_row[j] = (v1 += src_row[j]) + prev_sum_row[j];
                sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                sum_row[j + 3] = (v4 += src_row[j + 3]) + prev_sum_row[j + 3];
            }
        }
    } else {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    return CV_HAL_ERROR_OK;
}

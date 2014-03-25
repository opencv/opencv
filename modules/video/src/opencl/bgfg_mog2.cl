#if CN==1

#define T_MEAN float
#define F_ZERO (0.0f)
#define cnMode 1

#define frameToMean(a, b) (b) = *(a);
#define meanToFrame(a, b) *b = convert_uchar_sat(a);

inline float sqr(float val)
{
    return val * val;
}

inline float sum(float val)
{
    return val;
}

#else

#define T_MEAN float4
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)
#define cnMode 4

#define meanToFrame(a, b)\
    b[0] = convert_uchar_sat(a.x); \
    b[1] = convert_uchar_sat(a.y); \
    b[2] = convert_uchar_sat(a.z);

#define frameToMean(a, b)\
    b.x = a[0]; \
    b.y = a[1]; \
    b.z = a[2]; \
    b.w = 0.0f;

inline float sqr(const float4 val)
{
    return val.x * val.x + val.y * val.y + val.z * val.z;
}

inline float sum(const float4 val)
{
    return (val.x + val.y + val.z);
}

inline void swap4(__global float4* ptr, int x, int y, int k, int rows, int ptr_step)
{
    float4 val = ptr[(k * rows + y) * ptr_step + x];
    ptr[(k * rows + y) * ptr_step + x] = ptr[((k + 1) * rows + y) * ptr_step + x];
    ptr[((k + 1) * rows + y) * ptr_step + x] = val;
}

#endif

inline void swap(__global float* ptr, int x, int y, int k, int rows, int ptr_step)
{
    float val = ptr[(k * rows + y) * ptr_step + x];
    ptr[(k * rows + y) * ptr_step + x] = ptr[((k + 1) * rows + y) * ptr_step + x];
    ptr[((k + 1) * rows + y) * ptr_step + x] = val;
}

__kernel void mog2_kernel(__global const uchar* frame, int frame_step, int frame_offset, int frame_row, int frame_col, //uchar || uchar3
                          __global uchar* modesUsed, int modesUsed_step, int modesUsed_offset,                         //int
                          __global uchar* weight, int weight_step, int weight_offset,                                  //float
                          __global uchar* mean, int mean_step, int mean_offset,                                        //T_MEAN=float || float4
                          __global uchar* variance, int var_step, int var_offset,                                      //float
                          __global uchar* fgmask, int fgmask_step, int fgmask_offset,                                  //int
                          float alphaT, float alpha1, float prune,
                          int detectShadows_flag,
                          float c_Tb, float c_TB, float c_Tg, float c_varMin,                     //constants
                          float c_varMax, float c_varInit, float c_tau, uchar c_shadowVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    weight_step/= sizeof(float);
    var_step   /= sizeof(float);
    mean_step  /= (sizeof(float)*cnMode);

    if( x < frame_col && y < frame_row)
    {
        __global const uchar* _frame = (frame + mad24( y, frame_step, x*CN + frame_offset));
        T_MEAN pix;
        frameToMean(_frame, pix);

        bool background = false; // true - the pixel classified as background

        bool fitsPDF = false; //if it remains zero a new GMM mode will be added

        __global int* _modesUsed = (__global int*)(modesUsed + mad24( y, modesUsed_step, x*(int)(sizeof(int))));
        int nmodes = _modesUsed[0];
        int nNewModes = nmodes; //current number of modes in GMM

        float totalWeight = 0.0f;

        __global float* _weight = (__global float*)(weight);
        __global float* _variance = (__global float*)(variance);
        __global T_MEAN* _mean = (__global T_MEAN*)(mean);

        for (int mode = 0; mode < nmodes; ++mode)
        {

            float c_weight = alpha1 * _weight[(mode * frame_row + y) * weight_step + x] + prune;
            int swap_count = 0;
            if (!fitsPDF)
            {
                float c_var = _variance[(mode * frame_row + y) * var_step + x];

                T_MEAN c_mean = _mean[(mode * frame_row + y) * mean_step + x];

                T_MEAN diff = c_mean - pix;
                float dist2 = sqr(diff);

                if (totalWeight < c_TB && dist2 < c_Tb * c_var)
                    background = true;

                if (dist2 < c_Tg * c_var)
                {
                    fitsPDF = true;
                    c_weight += alphaT;
                    float k = alphaT / c_weight;

                    _mean[(mode * frame_row + y) * mean_step + x] = c_mean - k * diff;

                    float varnew = c_var + k * (dist2 - c_var);
                    varnew = fmax(varnew, c_varMin);
                    varnew = fmin(varnew, c_varMax);

                    _variance[(mode * frame_row + y) * var_step + x] = varnew;
                    for (int i = mode; i > 0; --i)
                    {
                        if (c_weight < _weight[((i - 1) * frame_row + y) * weight_step + x])
                            break;
                        swap_count++;
                        swap(_weight, x, y, i - 1, frame_row, weight_step);
                        swap(_variance, x, y, i - 1, frame_row, var_step);
                        #if (CN==1)
                        swap(_mean, x, y, i - 1, frame_row, mean_step);
                        #else
                        swap4(_mean, x, y, i - 1, frame_row, mean_step);
                        #endif
                    }
                }
            } // !fitsPDF

            if (c_weight < -prune)
            {
                c_weight = 0.0f;
                nmodes--;
            }

            _weight[((mode - swap_count) * frame_row + y) * weight_step + x] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
        }

        totalWeight = 1.f / totalWeight;
        for (int mode = 0; mode < nmodes; ++mode)
            _weight[(mode * frame_row + y) * weight_step + x] *= totalWeight;

        nmodes = nNewModes;

        if (!fitsPDF)
        {
            int mode = nmodes == (NMIXTURES) ? (NMIXTURES) - 1 : nmodes++;

            if (nmodes == 1)
                _weight[(mode * frame_row + y) * weight_step + x] = 1.f;
            else
            {
                _weight[(mode * frame_row + y) * weight_step + x] = alphaT;

                for (int i = 0; i < nmodes - 1; ++i)
                    _weight[(i * frame_row + y) * weight_step + x] *= alpha1;
            }

            _mean[(mode * frame_row + y) * mean_step + x] = pix;
            _variance[(mode * frame_row + y) * var_step + x] = c_varInit;

            for (int i = nmodes - 1; i > 0; --i)
            {
                if (alphaT < _weight[((i - 1) * frame_row + y) * weight_step + x])
                    break;

                swap(_weight, x, y, i - 1, frame_row, weight_step);
                swap(_variance, x, y, i - 1, frame_row, var_step);
                #if (CN==1)
                swap(_mean, x, y, i - 1, frame_row, mean_step);
                #else
                swap4(_mean, x, y, i - 1, frame_row, mean_step);
                #endif
            }
        }

        _modesUsed[0] = nmodes;
        bool isShadow = false;
        if (detectShadows_flag && !background)
        {
            float tWeight = 0.0f;

            for (int mode = 0; mode < nmodes; ++mode)
            {
                T_MEAN c_mean = _mean[(mode * frame_row + y) * mean_step + x];

                T_MEAN pix_mean = pix * c_mean;

                float numerator = sum(pix_mean);
                float denominator = sqr(c_mean);

                if (denominator == 0)
                    break;

                if (numerator <= denominator && numerator >= c_tau * denominator)
                {
                    float a = numerator / denominator;

                    T_MEAN dD = a * c_mean - pix;

                    if (sqr(dD) < c_Tb * _variance[(mode * frame_row + y) * var_step + x] * a * a)
                    {
                        isShadow = true;
                        break;
                    }
                }

                tWeight += _weight[(mode * frame_row + y) * weight_step + x];
                if (tWeight > c_TB)
                    break;
            }
        }
        __global int* _fgmask = (__global int*)(fgmask + mad24(y, fgmask_step, x*(int)(sizeof(int)) + fgmask_offset));
        *_fgmask = background ? 0 : isShadow ? c_shadowVal : 255;
    }
}

__kernel void getBackgroundImage2_kernel(__global const uchar* modesUsed, int modesUsed_step, int modesUsed_offset, int modesUsed_row, int modesUsed_col,
                                         __global const uchar* weight, int weight_step, int weight_offset,
                                         __global const uchar* mean, int mean_step, int mean_offset,
                                         __global uchar* dst, int dst_step, int dst_offset,
                                         float c_TB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < modesUsed_col && y < modesUsed_row)
    {
        __global int* _modesUsed = (__global int*)(modesUsed + mad24( y, modesUsed_step, x*(int)(sizeof(int))));
        int nmodes = _modesUsed[0];

        T_MEAN meanVal = (T_MEAN)F_ZERO;

        float totalWeight = 0.0f;

        for (int mode = 0; mode < nmodes; ++mode)
        {
            __global const float* _weight = (__global const float*)(weight + mad24(mode * modesUsed_row + y, weight_step, x*(int)(sizeof(float))));
            float c_weight = _weight[0];

            __global const T_MEAN* _mean = (__global const T_MEAN*)(mean + mad24(mode * modesUsed_row + y, mean_step, x*(int)(sizeof(float))*cnMode));
            T_MEAN c_mean = _mean[0];
            meanVal = meanVal + c_weight * c_mean;

            totalWeight += c_weight;

            if(totalWeight > c_TB)
                break;
        }

        meanVal = meanVal * (1.f / totalWeight);
        __global uchar* _dst = dst + y * dst_step + x*CN + dst_offset;
        meanToFrame(meanVal, _dst);
    }
}
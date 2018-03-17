#if CN==1

#define T_MEAN float
#define F_ZERO (0.0f)
#define cnMode 1

#define frameToMean(a, b) (b) = *(a);
#if FL==0
#define meanToFrame(a, b) *b = convert_uchar_sat(a);
#else
#define meanToFrame(a, b) *b = (float)a;
#endif

#else

#define T_MEAN float4
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)
#define cnMode 4

#if FL == 0
#define meanToFrame(a, b)\
    b[0] = convert_uchar_sat(a.x); \
    b[1] = convert_uchar_sat(a.y); \
    b[2] = convert_uchar_sat(a.z);
#else
#define meanToFrame(a, b)\
    b[0] = a.x; \
    b[1] = a.y; \
    b[2] = a.z;
#endif

#define frameToMean(a, b)\
    b.x = a[0]; \
    b.y = a[1]; \
    b.z = a[2]; \
    b.w = 0.0f;

#endif

__kernel void mog2_kernel(__global const uchar* frame, int frame_step, int frame_offset, int frame_row, int frame_col,  //uchar || uchar3
                          __global uchar* modesUsed,                                                                    //uchar
                          __global uchar* weight,                                                                       //float
                          __global uchar* mean,                                                                         //T_MEAN=float || float4
                          __global uchar* variance,                                                                     //float
                          __global uchar* fgmask, int fgmask_step, int fgmask_offset,                                   //uchar
                          float alphaT, float alpha1, float prune,
                          float c_Tb, float c_TB, float c_Tg, float c_varMin,                                           //constants
                          float c_varMax, float c_varInit, float c_tau
#ifdef SHADOW_DETECT
                          , uchar c_shadowVal
#endif
                          )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x < frame_col && y < frame_row)
    {
        #if FL==0
        __global const uchar* _frame = (frame + mad24(y, frame_step, mad24(x, CN, frame_offset)));
        #else
        __global const float* _frame = ((__global const float*)( frame + mad24(y, frame_step, frame_offset)) + mad24(x, CN, 0));
        #endif
        T_MEAN pix;
        frameToMean(_frame, pix);

        uchar foreground = 255; // 0 - the pixel classified as background

        bool fitsPDF = false; //if it remains zero a new GMM mode will be added

        int pt_idx =  mad24(y, frame_col, x);
        int idx_step = frame_row * frame_col;

        __global uchar* _modesUsed = modesUsed + pt_idx;
        uchar nmodes = _modesUsed[0];

        float totalWeight = 0.0f;

        __global float* _weight = (__global float*)(weight);
        __global float* _variance = (__global float*)(variance);
        __global T_MEAN* _mean = (__global T_MEAN*)(mean);

        uchar mode = 0;
        for (; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = mad(alpha1, _weight[mode_idx], prune);

            float c_var = _variance[mode_idx];

            T_MEAN c_mean = _mean[mode_idx];

            T_MEAN diff = c_mean - pix;
            float dist2 = dot(diff, diff);

            if (totalWeight < c_TB && dist2 < c_Tb * c_var)
                foreground = 0;

            if (dist2 < c_Tg * c_var)
            {
                fitsPDF = true;
                c_weight += alphaT;

                float k = alphaT / c_weight;
                T_MEAN mean_new = mad((T_MEAN)-k, diff, c_mean);
                float variance_new  = clamp(mad(k, (dist2 - c_var), c_var), c_varMin, c_varMax);

                for (int i = mode; i > 0; --i)
                {
                    int prev_idx = mode_idx - idx_step;
                    if (c_weight < _weight[prev_idx])
                        break;

                    _weight[mode_idx]   = _weight[prev_idx];
                    _variance[mode_idx] = _variance[prev_idx];
                    _mean[mode_idx]     = _mean[prev_idx];

                    mode_idx = prev_idx;
                }

                _mean[mode_idx]     = mean_new;
                _variance[mode_idx] = variance_new;
                _weight[mode_idx]   = c_weight; //update weight by the calculated value

                totalWeight += c_weight;

                mode ++;

                break;
            }
            if (c_weight < -prune)
                c_weight = 0.0f;

            _weight[mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
        }

        for (; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = mad(alpha1, _weight[mode_idx], prune);

            if (c_weight < -prune)
            {
                c_weight = 0.0f;
                nmodes = mode;
                break;
            }
            _weight[mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
        }

        if (0.f < totalWeight)
        {
            totalWeight = 1.f / totalWeight;
            for (int mode = 0; mode < nmodes; ++mode)
                _weight[mad24(mode, idx_step, pt_idx)] *= totalWeight;
        }

        if (!fitsPDF)
        {
            uchar mode = nmodes == (NMIXTURES) ? (NMIXTURES) - 1 : nmodes++;
            int mode_idx = mad24(mode, idx_step, pt_idx);

            if (nmodes == 1)
                _weight[mode_idx] = 1.f;
            else
            {
                _weight[mode_idx] = alphaT;

                for (int i = pt_idx; i < mode_idx; i += idx_step)
                    _weight[i] *= alpha1;
            }

            for (int i = nmodes - 1; i > 0; --i)
            {
                int prev_idx = mode_idx - idx_step;
                if (alphaT < _weight[prev_idx])
                    break;

                _weight[mode_idx]   = _weight[prev_idx];
                _variance[mode_idx] = _variance[prev_idx];
                _mean[mode_idx]     = _mean[prev_idx];

                mode_idx = prev_idx;
            }

            _mean[mode_idx] = pix;
            _variance[mode_idx] = c_varInit;
        }

        _modesUsed[0] = nmodes;
#ifdef SHADOW_DETECT
        if (foreground)
        {
            float tWeight = 0.0f;

            for (uchar mode = 0; mode < nmodes; ++mode)
            {
                int mode_idx = mad24(mode, idx_step, pt_idx);
                T_MEAN c_mean = _mean[mode_idx];

                float numerator = dot(pix, c_mean);
                float denominator = dot(c_mean, c_mean);

                if (denominator == 0)
                    break;

                if (numerator <= denominator && numerator >= c_tau * denominator)
                {
                    float a = numerator / denominator;

                    T_MEAN dD = mad(a, c_mean, -pix);

                    if (dot(dD, dD) < c_Tb * _variance[mode_idx] * a * a)
                    {
                        foreground = c_shadowVal;
                        break;
                    }
                }

                tWeight += _weight[mode_idx];
                if (tWeight > c_TB)
                    break;
            }
        }
#endif
        __global uchar* _fgmask = fgmask + mad24(y, fgmask_step, x + fgmask_offset);
        *_fgmask = (uchar)foreground;
    }
}

__kernel void getBackgroundImage2_kernel(__global const uchar* modesUsed,
                                         __global const uchar* weight,
                                         __global const uchar* mean,
                                         __global uchar* dst, int dst_step, int dst_offset, int dst_row, int dst_col,
                                         float c_TB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < dst_col && y < dst_row)
    {
        int pt_idx =  mad24(y, dst_col, x);

        __global const uchar* _modesUsed = modesUsed + pt_idx;
        uchar nmodes = _modesUsed[0];

        T_MEAN meanVal = (T_MEAN)F_ZERO;

        float totalWeight = 0.0f;
        __global const float* _weight = (__global const float*)weight;
        __global const T_MEAN* _mean = (__global const T_MEAN*)(mean);
        int idx_step = dst_row * dst_col;
        for (uchar mode = 0; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = _weight[mode_idx];
            T_MEAN c_mean = _mean[mode_idx];

            meanVal = mad(c_weight, c_mean, meanVal);

            totalWeight += c_weight;

            if (totalWeight > c_TB)
                break;
        }

        if (0.f < totalWeight)
            meanVal = meanVal / totalWeight;
        else
            meanVal = (T_MEAN)(0.f);

        #if FL==0
        __global uchar* _dst = dst + mad24(y, dst_step, mad24(x, CN, dst_offset));
        meanToFrame(meanVal, _dst);
        #else
        __global float* _dst = ((__global float*)( dst + mad24(y, dst_step, dst_offset)) + mad24(x, CN, 0));
        meanToFrame(meanVal, _dst);
        #endif
    }
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpConv_Winograd.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "convolution.hpp"

namespace cv { namespace dnn {

enum { VEC_ALIGN = 32, DFT_TYPE = CV_32F }; // Memory alignment.

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv,
                  int ntasks, float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct)
{
    const cv::dnn::Winofunc func =
        conv->useFP16 ? cv::dnn::getWinofunc_F16()
        : (conv->useAVX || conv->useAVX2 || conv->useNEON || conv->useRVV || conv->useSIMD128) ? cv::dnn::getWinofunc_F32()
        : cv::dnn::Winofunc::empty();

    if (!func.isGood())
        return 0;

    Mat input = _input.getMat();
    Mat output = _output.getMat();
    Mat fusedAddMat = _fusedAddMat.getMat();

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->K;
    int H0 = outputShape[2], W0 = outputShape[3];

    int pad_top = conv->pad_top;
    int pad_left = conv->pad_left;

    int ngroups = conv->ngroups, Cg = C/ngroups, Kg = K/ngroups;

    const int CONV_WINO_KBLOCK = 4;
    const int CONV_WINO_IBLOCK = func.iblock;
    const int CONV_WINO_ATOM = func.natom;
    const int CONV_WINO_NATOMS = CONV_WINO_AREA / CONV_WINO_ATOM;
    const int esz = func.esz;

    int Kg_nblocks = (Kg + CONV_WINO_KBLOCK - 1)/CONV_WINO_KBLOCK;
    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;

    int blocks_per_row = (W0+CONV_WINO_STEP-1)/CONV_WINO_STEP;
    int blocks_per_plane = ((H0+CONV_WINO_STEP-1)/CONV_WINO_STEP)*blocks_per_row;
    int blocks_per_plane_aligned = ((blocks_per_plane +
                                     CONV_WINO_IBLOCK-1)/CONV_WINO_IBLOCK)*CONV_WINO_IBLOCK;

    size_t totalbufsize = (size_t)N*C*blocks_per_plane_aligned*CONV_WINO_AREA;

    AutoBuffer<char> _buf;
    _buf.allocate((totalbufsize + VEC_ALIGN) * esz);
    char* wbuf_all = alignPtr(_buf.data(), VEC_ALIGN * esz);

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();

    float* fusedAddPtr = fusedAddMat.empty() ? nullptr : fusedAddMat.ptr<float>();

    // Phase 1. compute forward Winograd transforms for all input blocks,
    // all input planes, all samples in the batch.
    // [TODO]: maybe, if there are too many input channels, it makes sense to
    // transform only part of input channels at once and then compute the partial
    // accumulated sums (i.e. update the output buffers several times,
    // rather than compute them in one pass).
    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        int nc0 = (N*C)*task_id/ntasks;
        int nc1 = (N*C)*(task_id+1)/ntasks;
        for(; nc0 < nc1; nc0++)
        {
            int n = nc0 / C;
            int c = nc0 - n*C;
            int g = c / Cg;
            c -= g*Cg;

            for (int block_id = 0; block_id < blocks_per_plane; block_id += CONV_WINO_IBLOCK)
            {
                for (int db = 0; db < CONV_WINO_IBLOCK; db++)
                {
                    size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned +
                                     block_id)*Cg*CONV_WINO_AREA +
                                    (c*CONV_WINO_IBLOCK + db) * CONV_WINO_ATOM;
                    char* inwptr = wbuf_all + inwofs * esz;

                    if (block_id + db < blocks_per_plane)
                    {
                        int y0 = (block_id + db) / blocks_per_row;
                        int x0 = (block_id + db) - y0 * blocks_per_row;
                        y0 = y0*CONV_WINO_STEP - pad_top;
                        x0 = x0*CONV_WINO_STEP - pad_left;
                        bool partial = y0 < 0 || y0 + CONV_WINO_SIZE > Hi ||
                                       x0 < 0 || x0 + CONV_WINO_SIZE > Wi;
                        int dx1 = 0, dx2 = CONV_WINO_SIZE, dy1 = 0, dy2 = CONV_WINO_SIZE;
                        int inpstep = Wi;

                        float inpbuf[CONV_WINO_AREA];
                        float* inptr0 = (float*)inp + nc0*inp_planesize + y0*Wi + x0;
                        float* inptr = inptr0;

                        if (partial)
                        {
                            memset(inpbuf, 0, sizeof(inpbuf));
                            dy1 = -y0 > 0 ? -y0 : 0;
                            dy2 = Hi - y0 < CONV_WINO_SIZE ? Hi - y0 : CONV_WINO_SIZE;

                            if (dy2 < dy1) {dy2 = dy1 = 0;}
                            dx1 = -x0 > 0 ? -x0 : 0;
                            dx2 = Wi - x0 < CONV_WINO_SIZE ? Wi - x0 : CONV_WINO_SIZE;

                            if (dx2 < dx1) {dx2 = dx1 = 0;}
                            inptr0 -= y0*Wi + x0;

                            if (dx1 < dx2 && dy1 < dy2)
                            {
                                for(int dy = dy1; dy < dy2; dy++)
                                    memcpy(&inpbuf[dy*CONV_WINO_SIZE + dx1],
                                           inptr0 + (y0+dy)*Wi + (x0+dx1),
                                           (dx2-dx1)*sizeof(inpbuf[0]));
                            }

                            inptr = inpbuf;
                            inpstep = CONV_WINO_SIZE;
                        }
                        func.BtXB_8x8(inptr, inpstep, (uchar*)inwptr, Cg, CONV_WINO_IBLOCK, CONV_WINO_ATOM);
                    }
                    else
                    {
                        for (int i = 0; i < CONV_WINO_NATOMS; i++, inwptr += CONV_WINO_IBLOCK * CONV_WINO_ATOM * esz)
                            memset(inwptr, 0, CONV_WINO_ATOM * esz);
                    }
                }
            }
        }
    }});

    // Phase 2. compute elemwise-weighted sums of transformed blocks,
    // apply inverse Winograd transforms to the sums,
    // add bias, apply activation function if any and store the results.
    char* wptr0 = nullptr;
    if (esz == 2)
    {
        CV_Assert(!conv->weightsWinoBuf_FP16.empty());
        wptr0 = (char *)conv->getWeightsWinoFP16();
    }
    else if (esz == 4)
    {
        CV_Assert(!conv->weightsWinoBuf.empty());
        wptr0 = (char *)conv->getWeightsWino();
    }
    else
    {
        CV_Error(Error::StsError, "Impossible configuration");
    }

    parallel_for_(Range(0, ntasks), [&](const Range& r0) {
    for (int task_id = r0.start; task_id < r0.end; task_id++)
    {
        size_t out_wbuf_size = CONV_WINO_AREA * CONV_WINO_KBLOCK * CONV_WINO_IBLOCK;
        size_t outbuf_size = CONV_WINO_AREA;

        // For saving the accumulation output.
        AutoBuffer<char> out_wbuf_;
        out_wbuf_.allocate((out_wbuf_size + VEC_ALIGN) * esz);
        char* out_wbuf = alignPtr(out_wbuf_.data(), VEC_ALIGN * esz);
        memset(out_wbuf, 0, out_wbuf_size * esz);

        // For saving the fuse_Add data.
        AutoBuffer<float> outbuf_;
        outbuf_.allocate(outbuf_size + VEC_ALIGN);
        float* outbuf = alignPtr(outbuf_.data(), VEC_ALIGN);
        memset(outbuf, 0, outbuf_size * sizeof(outbuf[0]));

        int ngk0 = (int)(((int64_t)N*Kg_nblocks*ngroups)*task_id/ntasks);
        int ngk1 = (int)(((int64_t)N*Kg_nblocks*ngroups)*(task_id+1)/ntasks);

        for(; ngk0 < ngk1; ngk0++)
        {
            int n = ngk0 / (Kg_nblocks*ngroups);
            int gk0 = ngk0 % (Kg_nblocks*ngroups);
            int g = gk0 / Kg_nblocks;
            int k0 = (gk0 % Kg_nblocks)*CONV_WINO_KBLOCK;
            int k1 = k0 + CONV_WINO_KBLOCK <= Kg ? k0 + CONV_WINO_KBLOCK : Kg;

            for (int block_id0 = 0; block_id0 < blocks_per_plane; block_id0 += CONV_WINO_IBLOCK)
            {
                int block_id1 = block_id0 + CONV_WINO_IBLOCK;
                block_id1 = block_id1 < blocks_per_plane ? block_id1 : blocks_per_plane;
                size_t inwofs = ((n*ngroups + g)*blocks_per_plane_aligned + block_id0)*Cg*CONV_WINO_AREA;
                size_t wofs = (g*Kg_nblocks*CONV_WINO_KBLOCK + k0)*Cg*CONV_WINO_AREA;

                char* inwptr = wbuf_all + inwofs * esz;
                char* wptr = wptr0 + wofs * esz;

                func.accum((uchar*)inwptr, (uchar*)wptr, (uchar*)out_wbuf, Cg,
                           block_id1 - block_id0, CONV_WINO_IBLOCK,
                           CONV_WINO_KBLOCK, CONV_WINO_ATOM, CONV_WINO_NATOMS);

                for (int k = k0; k < k1; k++)
                {
                    float biasv = conv->biasBuf[g*Kg + k];
                    for (int block_id = block_id0; block_id < block_id1; block_id++)
                    {
                        int y0 = block_id / blocks_per_row;
                        int x0 = block_id - y0 * blocks_per_row;
                        y0 = y0*CONV_WINO_STEP;
                        x0 = x0*CONV_WINO_STEP;
                        int dy1 = H0 - y0;
                        if (dy1 > CONV_WINO_STEP) dy1 = CONV_WINO_STEP;
                        int dx1 = W0 - x0;
                        if (dx1 > CONV_WINO_STEP) dx1 = CONV_WINO_STEP;
                        assert(dx1 > 0 && dy1 > 0);
                        bool partial = activ || dy1 < CONV_WINO_STEP || dx1 < CONV_WINO_STEP;
                        size_t outofs = (n*K + g*Kg + k)*out_planesize + y0*W0 + x0;
                        int outstep = W0;

                        float* outptr0 = (float*)out + outofs;
                        float* pbptr0 = fusedAddPtr ? fusedAddPtr + outofs : nullptr;
                        float *outptr = outptr0, *bpptr = pbptr0;

                        if (partial)
                        {
                            outptr = outbuf;
                            outstep = CONV_WINO_SIZE;
                            if (pbptr0)
                            {
                                bpptr = outbuf;
                                for (int y = 0; y < dy1; y++)
                                    memcpy(outbuf + y*CONV_WINO_SIZE, pbptr0 + y*W0,
                                           dx1*sizeof(pbptr0[0]));
                            }
                        }

                        const int count = ((k - k0)*CONV_WINO_IBLOCK + (block_id - block_id0))*CONV_WINO_AREA;
                        func.AtXA_8x8((uchar*)out_wbuf + count * esz, CONV_WINO_SIZE,
                                      bpptr, outstep, outptr, outstep, biasv, minval, maxval, ifMinMaxAct);

                        if (partial)
                        {
                            if (activ)
                                activ->forwardSlice(outptr, outptr, CONV_WINO_SIZE*CONV_WINO_STEP, 0, g*Kg + k, g*Kg + k + 1);
                            for (int y = 0; y < dy1; y++)
                                memcpy(outptr0 + y*W0, outptr + y*CONV_WINO_SIZE, dx1*sizeof(outptr0[0]));
                        }
                    }
                }
            }
        }
    }});
    return 1;
}

}} // namespace cv::dnn

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/filesystem.private.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include "net_impl.hpp"
#include "op_migraphx.hpp"

#include <cstring>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_MIGRAPHX

static int migraphxTypeToCvDepth(migraphx_shape_datatype_t t)
{
    switch (t)
    {
        case migraphx_shape_float_type:  return CV_32F;
        case migraphx_shape_half_type:   return CV_16F;
        case migraphx_shape_double_type: return CV_64F;
        case migraphx_shape_int64_type:  return CV_64S;
        case migraphx_shape_int32_type:  return CV_32S;
        case migraphx_shape_int8_type:   return CV_8S;
        case migraphx_shape_uint8_type:  return CV_8U;
        default:                         return -1;
    }
}

// FNV-1a 64-bit: deterministic across processes (std::hash is not) -> stable cache key.
static uint64_t migraphxFnv1a(const void* data, size_t n, uint64_t h)
{
    const uchar* p = static_cast<const uchar*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= 1099511628211ULL; }
    return h;
}

// Path of the .mxr cache file for this (model, shapes, fp16, offload) combo, or empty
// when caching is disabled/unavailable. Mirrors the OpenVINO backend's getCacheDirectory
// + OPENCV_DNN_*_CACHE_DIR ("disabled") opt-out convention (see ie_ngraph.cpp).
static std::string migraphxCacheFile(const uchar* onnxData, size_t onnxSize,
                                     const std::vector<std::string>& inputNames,
                                     const std::vector<std::vector<int> >& inputShapes,
                                     bool fp16, bool offloadCopy)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    std::string dir = utils::fs::getCacheDirectory("dnn_migraphx_cache", "OPENCV_DNN_MIGRAPHX_CACHE_DIR");
#else
    std::string dir = utils::getConfigurationParameterString("OPENCV_DNN_MIGRAPHX_CACHE_DIR", "");
#endif
    if (dir.empty() || dir == "disabled")
        return std::string();
    utils::fs::createDirectories(dir);

    // Metadata that (with the model bytes) fully determines the compiled program.
    std::string meta = format("mgx=%d.%d.%d|fp16=%d|offload=%d",
                              MIGRAPHX_VERSION_MAJOR, MIGRAPHX_VERSION_MINOR, MIGRAPHX_VERSION_PATCH,
                              fp16 ? 1 : 0, offloadCopy ? 1 : 0);
    for (size_t i = 0; i < inputNames.size(); ++i)
    {
        meta += "|" + inputNames[i] + ":";
        if (i < inputShapes.size())
            for (size_t j = 0; j < inputShapes[i].size(); ++j)
                meta += format("%d,", inputShapes[i][j]);
    }
    uint64_t h = migraphxFnv1a(onnxData, onnxSize, 1469598103934665603ULL);
    h = migraphxFnv1a(meta.data(), meta.size(), h);
    return utils::fs::join(dir, format("dnn_migraphx_%016llx.mxr", (unsigned long long)h));
}

bool MIGraphXNet::build(const uchar* onnxData, size_t onnxSize,
                        const std::vector<std::string>& inputNames,
                        const std::vector<std::vector<int> >& inputShapes,
                        bool fp16)
{
    const std::string cacheFile =
        migraphxCacheFile(onnxData, onnxSize, inputNames, inputShapes, fp16, offloadCopy);

    // Fast path: load a previously compiled program (skips the expensive parse + compile).
    if (!cacheFile.empty() && utils::fs::exists(cacheFile))
    {
        try
        {
            prog = migraphx::load(cacheFile.c_str());
            paramNames.clear();
            migraphx::program_parameter_shapes lps = prog.get_parameter_shapes();
            std::vector<const char*> lnames = lps.names();
            for (size_t i = 0; i < lnames.size(); ++i)
                paramNames.push_back(std::string(lnames[i]));
            if (!paramNames.empty())
            {
                compiled = true;
                CV_LOG_INFO(NULL, "DNN/MIGraphX: loaded compiled program from cache: " << cacheFile);
                return true;
            }
        }
        catch (const std::exception& e)
        {
            CV_LOG_WARNING(NULL, "DNN/MIGraphX: cache load failed (" << e.what() << "); recompiling");
        }
    }

    try
    {
        migraphx::onnx_options oopt;
        for (size_t i = 0; i < inputNames.size() && i < inputShapes.size(); ++i)
        {
            if (inputNames[i].empty() || inputShapes[i].empty())
                continue;
            std::vector<std::size_t> dims(inputShapes[i].begin(), inputShapes[i].end());
            oopt.set_input_parameter_shape(inputNames[i], dims);
        }
        prog = migraphx::parse_onnx_buffer(reinterpret_cast<const void*>(onnxData), onnxSize, oopt);
        if (fp16)
            migraphx::quantize_fp16(prog);
        migraphx::compile_options copt;
        copt.set_offload_copy(offloadCopy);
        prog.compile(migraphx::target("gpu"), copt);

        paramNames.clear();
        migraphx::program_parameter_shapes pshapes = prog.get_parameter_shapes();
        std::vector<const char*> names = pshapes.names();
        for (size_t i = 0; i < names.size(); ++i)
            paramNames.push_back(std::string(names[i]));

        compiled = true;
        if (!cacheFile.empty())
        {
            try
            {
                migraphx::save(prog, cacheFile.c_str());
                CV_LOG_INFO(NULL, "DNN/MIGraphX: cached compiled program to: " << cacheFile);
            }
            catch (const std::exception& e)
            {
                CV_LOG_WARNING(NULL, "DNN/MIGraphX: cache save failed: " << e.what());
            }
        }
        CV_LOG_INFO(NULL, "DNN/MIGraphX: compiled program with " << paramNames.size() << " input parameter(s)");
        return true;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "DNN/MIGraphX: build failed: " << e.what());
        compiled = false;
        return false;
    }
}

bool MIGraphXNet::run(const std::vector<Mat>& inputBlobs,
                      const std::vector<std::string>& inputNames,
                      std::vector<Mat>& outputBlobs)
{
    if (!compiled)
        return false;
    try
    {
        migraphx::program_parameter_shapes pshapes = prog.get_parameter_shapes();
        migraphx::program_parameters pp;

        std::vector<Mat> holder;   // keep continuous copies alive until eval() returns
        holder.reserve(paramNames.size());
        for (size_t i = 0; i < paramNames.size(); ++i)
        {
            // Bind by MIGraphX's own parameter name (source of truth): match the
            // OpenCV input by name, else fall back to positional order.
            const std::string& name = paramNames[i];
            if (name.empty())
                continue;
            int idx = -1;
            for (size_t j = 0; j < inputNames.size(); ++j)
                if (inputNames[j] == name) { idx = (int)j; break; }
            if (idx < 0 && i < inputBlobs.size())
                idx = (int)i;
            if (idx < 0)
                continue;
            Mat m = inputBlobs[idx];
            if (!m.isContinuous())
                m = m.clone();
            holder.push_back(m);
            migraphx::shape s = pshapes[name.c_str()];
            pp.add(name.c_str(), migraphx::argument(s, (void*)holder.back().data));
        }

        migraphx::arguments outs = prog.eval(pp);
        outputBlobs.resize(outs.size());
        for (size_t i = 0; i < outs.size(); ++i)
        {
            migraphx::argument a = outs[i];
            migraphx::shape s = a.get_shape();
            int depth = migraphxTypeToCvDepth(s.type());
            if (depth < 0)
            {
                CV_LOG_WARNING(NULL, "DNN/MIGraphX: unsupported output dtype");
                return false;
            }
            std::vector<std::size_t> lens = s.lengths();
            std::vector<int> dims(lens.begin(), lens.end());
            if (dims.empty())
                dims.push_back(1);
            Mat out((int)dims.size(), dims.data(), depth);
            std::memcpy(out.data, a.data(), s.elements() * (size_t)CV_ELEM_SIZE1(depth));
            outputBlobs[i] = out;
        }
        return true;
    }
    catch (const std::exception& e)
    {
        CV_LOG_WARNING(NULL, "DNN/MIGraphX: run failed: " << e.what());
        return false;
    }
}

// -------- Net::Impl hooks (new graph engine, whole-model bypass) --------

bool Net::Impl::finalizeMIGraphX()
{
    if (migraphxNet && migraphxNet->compiled)
        return true;
    if (onnxModelBuffer.empty())
    {
        CV_LOG_ONCE_WARNING(NULL, "DNN/MIGraphX: no retained ONNX buffer; cannot offload to MIGraphX");
        return false;
    }

    // Inputs live in the new graph engine's tensors (setInput -> setGraphInput).
    std::vector<std::string> inNames;
    std::vector<std::vector<int> > inShapes;
    if (mainGraph)
    {
        const std::vector<Arg>& gin = mainGraph->inputs();
        for (size_t i = 0; i < gin.size(); ++i)
        {
            inNames.push_back(argData(gin[i]).name);
            const Mat& t = argTensor(gin[i]);
            if (!t.empty())
                inShapes.push_back(std::vector<int>(t.size.p, t.size.p + t.dims));
            else
                inShapes.push_back(std::vector<int>());
        }
    }

    migraphxNet = makePtr<MIGraphXNet>();
    const bool fp16 = (preferableTarget == DNN_TARGET_CUDA_FP16);
    if (!migraphxNet->build(onnxModelBuffer.data(), onnxModelBuffer.size(), inNames, inShapes, fp16))
    {
        migraphxNet.release();
        return false;
    }
    return true;
}

void Net::Impl::runMIGraphX(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
    CV_UNUSED(inputs);
    std::vector<std::string> inNames;
    std::vector<Mat> inBlobs;
    if (mainGraph)
    {
        const std::vector<Arg>& gin = mainGraph->inputs();
        for (size_t i = 0; i < gin.size(); ++i)
        {
            inNames.push_back(argData(gin[i]).name);
            inBlobs.push_back(argTensor(gin[i]));
        }
    }

    std::vector<Mat> outMats;
    if (!migraphxNet || !migraphxNet->run(inBlobs, inNames, outMats))
        CV_Error(Error::StsError, "DNN/MIGraphX: inference failed");

    _InputArray::KindFlag outKind = outputs.kind();
    if (outKind == _InputArray::STD_VECTOR_MAT)
    {
        outputs.getMatVecRef() = outMats;
    }
    else if (outKind == _InputArray::STD_VECTOR_UMAT)
    {
        std::vector<UMat>& v = outputs.getUMatVecRef();
        v.resize(outMats.size());
        for (size_t i = 0; i < outMats.size(); ++i)
            outMats[i].copyTo(v[i]);
    }
    else if (outKind == _InputArray::MAT || outKind == _InputArray::UMAT)
    {
        CV_CheckEQ((int)outMats.size(), 1, "DNN/MIGraphX: single Mat output requires exactly one program output");
        outMats[0].copyTo(outputs);
    }
    else
    {
        CV_Error(Error::StsBadArg, "DNN/MIGraphX: outputs must be Mat, UMat, or a vector of them");
    }
}

#endif // HAVE_MIGRAPHX

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/streaming/onevpl_cap.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser

const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help |                                    | Print this help message }"
    "{ input  |                                    | Path to the input video file }"
    "{ dec    | mjpeg or h265                      | Decoder type "
    ;

namespace detail {
typename cv::gapi::wip::CFGParams::value_type create_from_string(const std::string &line);
}

int main(int argc, char *argv[]) {

    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // get file name
    int param_index = 0;
    std::string file_path = cmd.get<std::string>("input");
    param_index++;

    // get VPL params from cmd
    cv::gapi::wip::CFGParams source_cfgs;
    try {
        do {
            std::string line = cmd.get<std::string>(param_index++);
            if (!cmd.check()) {
                break;
            }
            source_cfgs.insert(detail::create_from_string(line));
        } while(true);
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

    // Create source
    cv::Ptr<cv::gapi::wip::IStreamSource> cap =
            cv::gapi::wip::make_vpl_src(file_path, source_cfgs);
    
    try {
        cv::gapi::wip::Data data;

        bool status;
        do {
            status = cap->pull(data);
            cv::MediaFrame &frame = cv::util::get<cv::MediaFrame>(data);
            (void)frame;
            /* TODO cv::imshow("Out", frame);*/
        } while(status);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
    return 0;
}


namespace detail {
cv::gapi::wip::CFGParamValue create_cfg_impl_description(const std::string& value);
cv::gapi::wip::CFGParamValue create_cfg_decoder_codec_id(const std::string& value);
cv::gapi::wip::CFGParamValue create_cfg_accel_mode(const std::string& value);

typename cv::gapi::wip::CFGParams::value_type create_from_string(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CFGParams from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CFGParams from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 2);

    CFGParamValue candidate_value;
    if (name == "mfxImplDescription.Impl") {
        candidate_value = create_cfg_impl_description(value);
    } else if (name == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID") {
        candidate_value = create_cfg_decoder_codec_id(value);
    } else if (name == "mfxImplDescription.AccelerationMode") {
        candidate_value = create_cfg_accel_mode(value);
    } else {
        throw std::logic_error("Unhandled parameter name: " + name);
    }

    return {name, candidate_value};
}

cv::gapi::wip::CFGParamValue create_cfg_impl_description(const std::string& value) {
    cv::gapi::wip::CFGParamValue ret;
#ifdef HAVE_ONEVPL
    ret.Type = MFX_VARIANT_TYPE_U32;

    if (!value.compare("MFX_IMPL_TYPE_SOFTWARE")) {
        ret.Data.U32 = MFX_IMPL_TYPE_SOFTWARE;
    } else if (!value.compare("MFX_IMPL_TYPE_HARDWARE")) {
         ret.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    } else {
        throw std::logic_error("Cannot parse \"mfxImplDescription.Impl\" value: " + value);
    }
#endif
    return ret;
}

cv::gapi::wip::CFGParamValue create_cfg_decoder_codec_id(const std::string& value) {
    cv::gapi::wip::CFGParamValue ret;
#ifdef HAVE_ONEVPL
    ret.Type = MFX_VARIANT_TYPE_U32;

    if (!value.compare("MFX_CODEC_AVC")) {
        ret.Data.U32 = MFX_CODEC_AVC;
    } else if (!value.compare("MFX_CODEC_HEVC")) {
         ret.Data.U32 = MFX_CODEC_HEVC;
    } else if (!value.compare("MFX_CODEC_MPEG2")) {
         ret.Data.U32 = MFX_CODEC_MPEG2;
    } else if (!value.compare("MFX_CODEC_VC1")) {
         ret.Data.U32 = MFX_CODEC_VC1;
    } else if (!value.compare("MFX_CODEC_CAPTURE")) {
         ret.Data.U32 = MFX_CODEC_CAPTURE;
    } else if (!value.compare("MFX_CODEC_VP9")) {
         ret.Data.U32 = MFX_CODEC_VP9;
    } else if (!value.compare("MFX_CODEC_AV1")) {
         ret.Data.U32 = MFX_CODEC_AV1;
    } else {
        throw std::logic_error("Cannot parse \"mfxImplDescription.Impl\" value: " + value);
    }
#endif
    return ret;
}


cv::gapi::wip::CFGParamValue create_cfg_accel_mode(const std::string& value) {
    cv::gapi::wip::CFGParamValue ret;
#ifdef HAVE_ONEVPL
    ret.Type = MFX_VARIANT_TYPE_U32;

    if (!value.compare("MFX_ACCEL_MODE_NA")) {
        ret.Data.U32 = MFX_ACCEL_MODE_NA;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_D3D9")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_D3D9;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_D3D11")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_VAAPI")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_VAAPI_GLX")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_GLX;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_VAAPI_X11")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_X11;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND;
    } else if (!value.compare("MFX_ACCEL_MODE_VIA_HDDLUNITE")) {
         ret.Data.U32 = MFX_ACCEL_MODE_VIA_HDDLUNITE;
    } else {
        throw std::logic_error("Cannot parse \"mfxImplDescription.Impl\" value: " + value);
    }
#endif
    return ret;
}
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAMS_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAMS_HPP

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/util/variant.hpp>

namespace cv {
namespace gapi {
namespace wip {

struct GAPI_EXPORTS oneVPL_cfg_param {
    using name_t = std::string;
    using value_t = cv::util::variant<uint8_t, int8_t,
                                      uint16_t, int16_t,
                                      uint32_t, int32_t,
                                      uint64_t, int64_t,
                                      float_t,
                                      double_t,
                                      void*,
                                      std::string>;
    template<typename ValueType>
    static oneVPL_cfg_param create(const std::string& name, ValueType&& value, bool is_major = true) {
        oneVPL_cfg_param param(name, oneVPL_cfg_param::value_t(std::forward<ValueType>(value)), is_major);
        return param;
    }

    struct Priv;

    const name_t& get_name() const;
    const value_t& get_value() const;
    bool is_major() const;
    bool operator==(const oneVPL_cfg_param& src) const;
    bool operator< (const oneVPL_cfg_param& src) const;
    bool operator!=(const oneVPL_cfg_param& src) const;

    oneVPL_cfg_param& operator=(const oneVPL_cfg_param& src);
    oneVPL_cfg_param& operator=(oneVPL_cfg_param&& src);
    oneVPL_cfg_param(const oneVPL_cfg_param& src);
    oneVPL_cfg_param(oneVPL_cfg_param&& src);
    ~oneVPL_cfg_param();
private:
    oneVPL_cfg_param(const std::string& param_name, value_t&& param_value, bool is_major_param);
    std::shared_ptr<Priv> m_priv;
};

} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_ONEVPL_CFG_PARAMS_HPP

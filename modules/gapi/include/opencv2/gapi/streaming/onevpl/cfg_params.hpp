// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/util/variant.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS CfgParam {
    using name_t = std::string;
    using value_t = cv::util::variant<uint8_t, int8_t,
                                      uint16_t, int16_t,
                                      uint32_t, int32_t,
                                      uint64_t, int64_t,
                                      float_t,
                                      double_t,
                                      void*,
                                      std::string>;

    /**
     * Create onevpl source configuration parameter.
     *
     *@param name           name of parameter.
     *@param value          value of parameter.
     *@param is_major       TRUE if parameter MUST be provided by OneVPL implementation, FALSE for optional (for resolve multiple available implementations).
     *
     */
    template<typename ValueType>
    static CfgParam create(const std::string& name, ValueType&& value, bool is_major = true) {
        CfgParam param(name, CfgParam::value_t(std::forward<ValueType>(value)), is_major);
        return param;
    }

    struct Priv;

    const name_t& get_name() const;
    const value_t& get_value() const;
    bool is_major() const;
    bool operator==(const CfgParam& rhs) const;
    bool operator< (const CfgParam& rhs) const;
    bool operator!=(const CfgParam& rhs) const;

    CfgParam& operator=(const CfgParam& src);
    CfgParam& operator=(CfgParam&& src);
    CfgParam(const CfgParam& src);
    CfgParam(CfgParam&& src);
    ~CfgParam();
private:
    CfgParam(const std::string& param_name, value_t&& param_value, bool is_major_param);
    std::shared_ptr<Priv> m_priv;
};

} //namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_CFG_PARAMS_HPP

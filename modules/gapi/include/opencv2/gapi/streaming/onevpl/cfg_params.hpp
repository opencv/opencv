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

/**
 * @brief Public class is using for creation of onevpl::GSource instances.
 *
 * Class members availaible through methods @ref CfgParam::get_name() and @ref CfgParam::get_value() are used by
 * onevpl::GSource inner logic to create or find oneVPL particular implementation
 * (software/hardware, specific API version and etc.).
 *
 * @note Because oneVPL may provide several implementations which are satisfying with multiple (or single one) @ref CfgParam
 * criteria therefore it is possible to configure `preferred` parameters. This kind of CfgParams are created
 * using `is_major = false` argument in @ref CfgParam::create method and are not used by creating oneVPL particular implementations.
 * Instead they fill out a "score table" to select preferrable implementation from available list. Implementation are satisfying
 * with most of these optional params would be chosen.
 * If no one optional CfgParam params were present then first of available oneVPL implementation would be applied.
 * Please get on https://spec.oneapi.io/versions/latest/elements/oneVPL/source/API_ref/VPL_disp_api_func.html?highlight=mfxcreateconfig#mfxsetconfigfilterproperty
 * for using OneVPL configuration. In this schema `mfxU8 *name` represents @ref CfgParam::get_name() and
 * `mfxVariant value` is @ref CfgParam::get_value()
 */
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
     * Create onevp::GSource configuration parameter.
     *
     *@param name           name of parameter.
     *@param value          value of parameter.
     *@param is_major       TRUE if parameter MUST be provided by OneVPL inner implementation, FALSE for optional (for resolve multiple available implementations).
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

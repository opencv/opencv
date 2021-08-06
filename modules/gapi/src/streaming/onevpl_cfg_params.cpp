// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/util/throw.hpp>

#include <opencv2/gapi/streaming/onevpl_cfg_params.hpp>

namespace cv {
namespace gapi {
namespace wip {

struct oneVPL_cfg_param::Priv {
    Priv(const std::string& param_name, oneVPL_cfg_param::value_t&& param_value, bool is_major_param) :
        name(param_name), value(std::forward<value_t>(param_value)), major_flag(is_major_param) {
    }

    const oneVPL_cfg_param::name_t& get_name_impl() const {
        return name;
    }

    const oneVPL_cfg_param::value_t& get_value_impl() const {
        return value;
    }

    bool is_major_impl() const {
        return major_flag;
    }

    oneVPL_cfg_param::name_t name;
    oneVPL_cfg_param::value_t value;
    bool major_flag;
};

oneVPL_cfg_param::oneVPL_cfg_param (const std::string& param_name, value_t&& param_value, bool is_major_param) :
    m_priv(new Priv(param_name, std::move(param_value), is_major_param)) {
}

oneVPL_cfg_param::~oneVPL_cfg_param() {
}

oneVPL_cfg_param& oneVPL_cfg_param::operator=(const oneVPL_cfg_param& src) {
    if (this != &src) {
        m_priv = src.m_priv;
    }
    return *this;
}

oneVPL_cfg_param& oneVPL_cfg_param::operator=(oneVPL_cfg_param&& src) {
    if (this != &src) {
        m_priv = std::move(src.m_priv);
    }
    return *this;
}

oneVPL_cfg_param::oneVPL_cfg_param(const oneVPL_cfg_param& src) :
    m_priv(src.m_priv) {
}

oneVPL_cfg_param::oneVPL_cfg_param(oneVPL_cfg_param&& src) :
    m_priv(std::move(src.m_priv)) {
}

const oneVPL_cfg_param::name_t& oneVPL_cfg_param::get_name() const {
    return m_priv->get_name_impl();
}

const oneVPL_cfg_param::value_t& oneVPL_cfg_param::get_value() const {
    return m_priv->get_value_impl();
}

bool oneVPL_cfg_param::is_major() const {
    return m_priv->is_major_impl();
}

struct variant_comparator : cv::util::static_visitor<bool, variant_comparator> {
    variant_comparator(const oneVPL_cfg_param::value_t& rhs_value) :
        rhs(rhs_value) {}

    template<typename ValueType>
    bool visit(const ValueType& lhs) {
        return lhs < cv::util::get<ValueType>(rhs);
    }
private:
    const oneVPL_cfg_param::value_t& rhs;
};

bool oneVPL_cfg_param::operator< (const oneVPL_cfg_param& src) const {
    // implement default pair comparison
    if (get_name() < src.get_name()) {
        return true;
    } else if (get_name() > src.get_name()) {
        return false;
    }

    //TODO implement operator < for cv::util::variant
    const oneVPL_cfg_param::value_t& lvar = get_value();
    const oneVPL_cfg_param::value_t& rvar = src.get_value();
    if (lvar.index() < rvar.index()) {
        return true;
    } else if (lvar.index() > rvar.index()) {
        return false;
    }

    variant_comparator comp(rvar);
    return cv::util::visit(comp, lvar);
}

bool oneVPL_cfg_param::operator==(const oneVPL_cfg_param& src) const {
    return (get_name() == src.get_name()) && (get_value() == src.get_value());
}

bool oneVPL_cfg_param::operator!=(const oneVPL_cfg_param& src) const {
    return !(*this == src);
}

} // namespace wip
} // namespace gapi
} // namespace cv

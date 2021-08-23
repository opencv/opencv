// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/util/throw.hpp>

#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

namespace util {
struct variant_comparator : cv::util::static_visitor<bool, variant_comparator> {
    variant_comparator(const CfgParam::value_t& rhs_value) :
        rhs(rhs_value) {}

    template<typename ValueType>
    bool visit(const ValueType& lhs) const {
        return lhs < cv::util::get<ValueType>(rhs);
    }
private:
    const CfgParam::value_t& rhs;
};
} // namespace util

struct CfgParam::Priv {
    Priv(const std::string& param_name, CfgParam::value_t&& param_value, bool is_major_param) :
        name(param_name), value(std::forward<value_t>(param_value)), major_flag(is_major_param) {
    }

    const CfgParam::name_t& get_name_impl() const {
        return name;
    }

    const CfgParam::value_t& get_value_impl() const {
        return value;
    }

    bool is_major_impl() const {
        return major_flag;
    }

    // comparison implementation
    bool operator< (const Priv& rhs) const {
        // implement default pair comparison
        if (get_name_impl() < rhs.get_name_impl()) {
            return true;
        } else if (get_name_impl() > rhs.get_name_impl()) {
            return false;
        }

        //TODO implement operator < for cv::util::variant
        const CfgParam::value_t& lvar = get_value_impl();
        const CfgParam::value_t& rvar = rhs.get_value_impl();
        if (lvar.index() < rvar.index()) {
            return true;
        } else if (lvar.index() > rvar.index()) {
            return false;
        }

        util::variant_comparator comp(rvar);
        return cv::util::visit(comp, lvar);
    }

    bool operator==(const Priv& rhs) const {
        return (get_name_impl() == rhs.get_name_impl())
                && (get_value_impl() == rhs.get_value_impl());
    }

    bool operator!=(const Priv& rhs) const {
        return !(*this == rhs);
    }

    CfgParam::name_t name;
    CfgParam::value_t value;
    bool major_flag;
};

CfgParam::CfgParam (const std::string& param_name, value_t&& param_value, bool is_major_param) :
    m_priv(new Priv(param_name, std::move(param_value), is_major_param)) {
}

CfgParam::~CfgParam() = default;

CfgParam& CfgParam::operator=(const CfgParam& src) {
    if (this != &src) {
        m_priv = src.m_priv;
    }
    return *this;
}

CfgParam& CfgParam::operator=(CfgParam&& src) {
    if (this != &src) {
        m_priv = std::move(src.m_priv);
    }
    return *this;
}

CfgParam::CfgParam(const CfgParam& src) :
    m_priv(src.m_priv) {
}

CfgParam::CfgParam(CfgParam&& src) :
    m_priv(std::move(src.m_priv)) {
}

const CfgParam::name_t& CfgParam::get_name() const {
    return m_priv->get_name_impl();
}

const CfgParam::value_t& CfgParam::get_value() const {
    return m_priv->get_value_impl();
}

bool CfgParam::is_major() const {
    return m_priv->is_major_impl();
}

bool CfgParam::operator< (const CfgParam& rhs) const {
    return *m_priv < *rhs.m_priv;
}

bool CfgParam::operator==(const CfgParam& rhs) const {
    return *m_priv == *rhs.m_priv;
}

bool CfgParam::operator!=(const CfgParam& rhs) const {
    return *m_priv != *rhs.m_priv;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

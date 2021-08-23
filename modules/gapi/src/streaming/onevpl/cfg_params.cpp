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
    variant_comparator(const cfg_param::value_t& rhs_value) :
        rhs(rhs_value) {}

    template<typename ValueType>
    bool visit(const ValueType& lhs) {
        return lhs < cv::util::get<ValueType>(rhs);
    }
private:
    const cfg_param::value_t& rhs;
};
} // namespace util

struct cfg_param::Priv {
    Priv(const std::string& param_name, cfg_param::value_t&& param_value, bool is_major_param) :
        name(param_name), value(std::forward<value_t>(param_value)), major_flag(is_major_param) {
    }

    const cfg_param::name_t& get_name_impl() const {
        return name;
    }

    const cfg_param::value_t& get_value_impl() const {
        return value;
    }

    bool is_major_impl() const {
        return major_flag;
    }

    // comparison implementation
    bool operator< (const Priv& src) const {
        // implement default pair comparison
        if (get_name_impl() < src.get_name_impl()) {
            return true;
        } else if (get_name_impl() > src.get_name_impl()) {
            return false;
        }

        //TODO implement operator < for cv::util::variant
        const cfg_param::value_t& lvar = get_value_impl();
        const cfg_param::value_t& rvar = src.get_value_impl();
        if (lvar.index() < rvar.index()) {
            return true;
        } else if (lvar.index() > rvar.index()) {
            return false;
        }

        util::variant_comparator comp(rvar);
        return cv::util::visit(comp, lvar);
    }

    bool operator==(const Priv& src) const {
        return (get_name_impl() == src.get_name_impl())
                && (get_value_impl() == src.get_value_impl());
    }

    bool operator!=(const Priv& src) const {
        return !(*this == src);
    }

    cfg_param::name_t name;
    cfg_param::value_t value;
    bool major_flag;
};

cfg_param::cfg_param (const std::string& param_name, value_t&& param_value, bool is_major_param) :
    m_priv(new Priv(param_name, std::move(param_value), is_major_param)) {
}

cfg_param::~cfg_param() = default;

cfg_param& cfg_param::operator=(const cfg_param& src) {
    if (this != &src) {
        m_priv = src.m_priv;
    }
    return *this;
}

cfg_param& cfg_param::operator=(cfg_param&& src) {
    if (this != &src) {
        m_priv = std::move(src.m_priv);
    }
    return *this;
}

cfg_param::cfg_param(const cfg_param& src) :
    m_priv(src.m_priv) {
}

cfg_param::cfg_param(cfg_param&& src) :
    m_priv(std::move(src.m_priv)) {
}

const cfg_param::name_t& cfg_param::get_name() const {
    return m_priv->get_name_impl();
}

const cfg_param::value_t& cfg_param::get_value() const {
    return m_priv->get_value_impl();
}

bool cfg_param::is_major() const {
    return m_priv->is_major_impl();
}

bool cfg_param::operator< (const cfg_param& src) const {
    return *m_priv < *src.m_priv;
}

bool cfg_param::operator==(const cfg_param& src) const {
    return *m_priv == *src.m_priv;
}

bool cfg_param::operator!=(const cfg_param& src) const {
    return *m_priv != *src.m_priv;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

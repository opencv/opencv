#include <opencv2/gapi/infer/bindings_ie.hpp>

cv::gapi::ie::PyParams::PyParams(const std::string &tag,
                                 const std::string &model,
                                 const std::string &weights,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<cv::gapi::Generic>>(tag, model, weights, device)) {
}

cv::gapi::ie::PyParams::PyParams(const std::string &tag,
                                 const std::string &model,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<cv::gapi::Generic>>(tag, model, device)) {
}

cv::gapi::GBackend cv::gapi::ie::PyParams::backend() const {
    return m_priv->backend();
}

std::string cv::gapi::ie::PyParams::tag() const {
    return m_priv->tag();
}

cv::util::any cv::gapi::ie::PyParams::params() const {
    return m_priv->params();
}

cv::gapi::ie::PyParams cv::gapi::ie::params(const std::string &tag,
                                            const std::string &model,
                                            const std::string &weights,
                                            const std::string &device) {
    return {tag, model, weights, device};
}

cv::gapi::ie::PyParams cv::gapi::ie::params(const std::string &tag,
                                            const std::string &model,
                                            const std::string &device) {
    return {tag, model, device};
}

cv::gapi::ie::PyParams& cv::gapi::ie::PyParams::constInput(const std::string &layer_name,
                                                           const cv::Mat &data,
                                                           TraitAs hint) {
    m_priv->constInput(layer_name, data, hint);
    return *this;
}

cv::gapi::ie::PyParams& cv::gapi::ie::PyParams::cfgNumRequests(size_t nireq) {
    m_priv->cfgNumRequests(nireq);
    return *this;
}

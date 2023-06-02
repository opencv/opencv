#include <opencv2/gapi/infer/bindings_ov.hpp>

cv::gapi::ov::PyParams::PyParams(const std::string &tag,
                                 const std::string &model_path,
                                 const std::string &bin_path,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<cv::gapi::Generic>>(tag, model_path, bin_path, device)) {
}

cv::gapi::ov::PyParams::PyParams(const std::string &tag,
                                 const std::string &blob_path,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<cv::gapi::Generic>>(tag, blob_path, device)) {
}

cv::gapi::GBackend cv::gapi::ov::PyParams::backend() const {
    return m_priv->backend();
}

std::string cv::gapi::ov::PyParams::tag() const {
    return m_priv->tag();
}

cv::util::any cv::gapi::ov::PyParams::params() const {
    return m_priv->params();
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgPluginConfig(
        const std::map<std::string, std::string> &config) {
    m_priv->cfgPluginConfig(config);
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgInputTensorLayout(std::string tensor_layout) {
    m_priv->cfgInputTensorLayout(std::move(tensor_layout));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgInputTensorLayout(
        std::map<std::string, std::string> layout_map) {
    m_priv->cfgInputTensorLayout(std::move(layout_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgInputModelLayout(std::string tensor_layout) {
    m_priv->cfgInputModelLayout(std::move(tensor_layout));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgInputModelLayout(
        std::map<std::string, std::string> layout_map) {
    m_priv->cfgInputModelLayout(std::move(layout_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputTensorLayout(std::string tensor_layout) {
    m_priv->cfgOutputTensorLayout(std::move(tensor_layout));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputTensorLayout(
        std::map<std::string, std::string> layout_map) {
    m_priv->cfgOutputTensorLayout(std::move(layout_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputModelLayout(std::string tensor_layout) {
    m_priv->cfgOutputModelLayout(std::move(tensor_layout));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputModelLayout(
        std::map<std::string, std::string> layout_map) {
    m_priv->cfgOutputModelLayout(std::move(layout_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputTensorPrecision(int precision) {
    m_priv->cfgOutputTensorPrecision(precision);
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgOutputTensorPrecision(
        std::map<std::string, int> precision_map) {
    m_priv->cfgOutputTensorPrecision(precision_map);
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgReshape(std::vector<size_t> new_shape) {
    m_priv->cfgReshape(std::move(new_shape));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgReshape(
        std::map<std::string, std::vector<size_t>> new_shape_map) {
    m_priv->cfgReshape(std::move(new_shape_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgNumRequests(const size_t nireq) {
    m_priv->cfgNumRequests(nireq);
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgMean(std::vector<float> mean_values) {
    m_priv->cfgMean(std::move(mean_values));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgMean(
        std::map<std::string, std::vector<float>> mean_map) {
    m_priv->cfgMean(std::move(mean_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgScale(std::vector<float> scale_values) {
    m_priv->cfgScale(std::move(scale_values));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgScale(
        std::map<std::string, std::vector<float>> scale_map) {
    m_priv->cfgScale(std::move(scale_map));
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgResize(int interpolation) {
    m_priv->cfgResize(interpolation);
    return *this;
}

cv::gapi::ov::PyParams&
cv::gapi::ov::PyParams::cfgResize(std::map<std::string, int> interpolation) {
    m_priv->cfgResize(std::move(interpolation));
    return *this;
}

cv::gapi::ov::PyParams cv::gapi::ov::params(const std::string &tag,
                                            const std::string &model_path,
                                            const std::string &weights,
                                            const std::string &device) {
    return {tag, model_path, weights, device};
}

cv::gapi::ov::PyParams cv::gapi::ov::params(const std::string &tag,
                                            const std::string &blob_path,
                                            const std::string &device) {
    return {tag, blob_path, device};
}

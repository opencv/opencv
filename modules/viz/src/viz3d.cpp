#include <opencv2/viz/viz3d.hpp>
#include "viz3d_impl.hpp"


cv::viz::Viz3d::Viz3d(const String& window_name) : impl_(new VizImpl(window_name)) {}
cv::viz::Viz3d::~Viz3d() { delete impl_; }

void cv::viz::Viz3d::setBackgroundColor(const Color& color) { impl_->setBackgroundColor(color); }

bool cv::viz::Viz3d::addPolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->addPolygonMesh(mesh, Mat(), id);
}

bool cv::viz::Viz3d::updatePolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->updatePolygonMesh(mesh, Mat(), id);
}

bool cv::viz::Viz3d::addPolylineFromPolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->addPolylineFromPolygonMesh(mesh, id);
}

bool cv::viz::Viz3d::addPolygon(const Mat& cloud, const Color& color, const String& id)
{
    return impl_->addPolygon(cloud, color, id);
}

void cv::viz::Viz3d::spin() { impl_->spin(); }
void cv::viz::Viz3d::spinOnce (int time, bool force_redraw) { impl_->spinOnce(time, force_redraw); }
bool cv::viz::Viz3d::wasStopped() const { return impl_->wasStopped(); }

void cv::viz::Viz3d::registerKeyboardCallback(KeyboardCallback callback, void* cookie)
{ impl_->registerKeyboardCallback(callback, cookie); }

void cv::viz::Viz3d::registerMouseCallback(MouseCallback callback, void* cookie)
{ impl_->registerMouseCallback(callback, cookie); }



void cv::viz::Viz3d::showWidget(const String &id, const Widget &widget, const Affine3f &pose) { impl_->showWidget(id, widget, pose); }
void cv::viz::Viz3d::removeWidget(const String &id) { impl_->removeWidget(id); }
cv::viz::Widget cv::viz::Viz3d::getWidget(const String &id) const { return impl_->getWidget(id); }
void cv::viz::Viz3d::setWidgetPose(const String &id, const Affine3f &pose) { impl_->setWidgetPose(id, pose); }
void cv::viz::Viz3d::updateWidgetPose(const String &id, const Affine3f &pose) { impl_->updateWidgetPose(id, pose); }
cv::Affine3f cv::viz::Viz3d::getWidgetPose(const String &id) const { return impl_->getWidgetPose(id); }

void cv::viz::Viz3d::setCamera(const Camera &camera) { impl_->setCamera(camera); }
cv::viz::Camera cv::viz::Viz3d::getCamera() const { return impl_->getCamera(); }
void cv::viz::Viz3d::setViewerPose(const Affine3f &pose) { impl_->setViewerPose(pose); }
cv::Affine3f cv::viz::Viz3d::getViewerPose() { return impl_->getViewerPose(); }

void cv::viz::Viz3d::convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord) { impl_->convertToWindowCoordinates(pt, window_coord); }
void cv::viz::Viz3d::converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction) { impl_->converTo3DRay(window_coord, origin, direction); }
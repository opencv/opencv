#include <opencv2/viz/viz3d.hpp>
#include <q/viz3d_impl.hpp>


temp_viz::Viz3d::Viz3d(const String& window_name) : impl_(new VizImpl(window_name))
{

}

temp_viz::Viz3d::~Viz3d()
{

}


void temp_viz::Viz3d::setBackgroundColor(const Color& color)
{
    impl_->setBackgroundColor(color);
}

void temp_viz::Viz3d::addCoordinateSystem(double scale, const Affine3f& t, const String &id)
{
    impl_->addCoordinateSystem(scale, t, id);
}

void temp_viz::Viz3d::showPointCloud(const String& id, InputArray cloud, InputArray colors, const Affine3f& pose)
{
    impl_->showPointCloud(id, cloud, colors, pose);
}

void temp_viz::Viz3d::showPointCloud(const String& id, InputArray cloud, const Color& color, const Affine3f& pose)
{
    impl_->showPointCloud(id, cloud, color, pose);
}

bool temp_viz::Viz3d::addPointCloudNormals (const Mat &cloud, const Mat& normals, int level, float scale, const String& id)
{
    return impl_->addPointCloudNormals(cloud, normals, level, scale, id);
}

bool temp_viz::Viz3d::addPolygonMesh (const Mesh3d& mesh, const String &id)
{
    return impl_->addPolygonMesh(mesh, Mat(), id);
}

bool temp_viz::Viz3d::updatePolygonMesh (const Mesh3d& mesh, const String &id)
{
    return impl_->updatePolygonMesh(mesh, Mat(), id);
}

bool temp_viz::Viz3d::addPolylineFromPolygonMesh (const Mesh3d& mesh, const String &id)
{
    return impl_->addPolylineFromPolygonMesh(mesh, id);
}

bool temp_viz::Viz3d::addText (const String &text, int xpos, int ypos, const Color& color, int fontsize, const String &id)
{
    return impl_->addText(text, xpos, ypos, color, fontsize, id);
}

bool temp_viz::Viz3d::addPolygon(const Mat& cloud, const Color& color, const String& id)
{
    return impl_->addPolygon(cloud, color, id);
}

bool temp_viz::Viz3d::addSphere (const cv::Point3f &center, double radius, const Color& color, const std::string &id)
{
    return impl_->addSphere(center, radius, color, id);
}

void temp_viz::Viz3d::spin()
{
    impl_->spin();
}

void temp_viz::Viz3d::spinOnce (int time, bool force_redraw)
{
    impl_->spinOnce(time, force_redraw);
}

void temp_viz::Viz3d::showLine(const String &id, const Point3f &pt1, const Point3f &pt2, const Color &color)
{
    impl_->showLine(id, pt1, pt2, color);
}

void temp_viz::Viz3d::showPlane(const String &id, const Vec4f &coefs, const Color &color)
{
    impl_->showPlane(id, coefs, color);
}

void temp_viz::Viz3d::showPlane(const String &id, const Vec4f &coefs, const Point3f &pt, const Color &color)
{
    impl_->showPlane(id, coefs, pt, color);
}

bool temp_viz::Viz3d::removeCoordinateSystem (const String &id)
{
    return impl_->removeCoordinateSystem(id);
}

void temp_viz::Viz3d::registerKeyboardCallback(void (*callback)(const cv::KeyboardEvent&, void*), void* cookie)
{
    impl_->registerKeyboardCallback(callback, cookie);
}

void temp_viz::Viz3d::registerMouseCallback(void (*callback)(const cv::MouseEvent&, void*), void* cookie)
{
    impl_->registerMouseCallback(callback, cookie);
}

bool temp_viz::Viz3d::wasStopped() const { return impl_->wasStopped(); }


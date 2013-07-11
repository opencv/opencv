#include <opencv2/viz/viz3d.hpp>
#include <q/viz3d_impl.hpp>


temp_viz::Viz3d::Viz3d(const String& window_name) : impl_(new VizImpl(window_name))
{

}

temp_viz::Viz3d::~Viz3d()
{
    delete impl_;
}

void temp_viz::Viz3d::setBackgroundColor(const Color& color)
{
    impl_->setBackgroundColor(color);
}

bool temp_viz::Viz3d::addPolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->addPolygonMesh(mesh, Mat(), id);
}

bool temp_viz::Viz3d::updatePolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->updatePolygonMesh(mesh, Mat(), id);
}

bool temp_viz::Viz3d::addPolylineFromPolygonMesh (const Mesh3d& mesh, const String& id)
{
    return impl_->addPolylineFromPolygonMesh(mesh, id);
}

bool temp_viz::Viz3d::addPolygon(const Mat& cloud, const Color& color, const String& id)
{
    return impl_->addPolygon(cloud, color, id);
}

void temp_viz::Viz3d::spin()
{
    impl_->spin();
}

void temp_viz::Viz3d::spinOnce (int time, bool force_redraw)
{
    impl_->spinOnce(time, force_redraw);
}

void temp_viz::Viz3d::registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void* cookie)
{
    impl_->registerKeyboardCallback(callback, cookie);
}

void temp_viz::Viz3d::registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie)
{
    impl_->registerMouseCallback(callback, cookie);
}

bool temp_viz::Viz3d::wasStopped() const { return impl_->wasStopped(); }

void temp_viz::Viz3d::showWidget(const String &id, const Widget &widget, const Affine3f &pose)
{
    impl_->showWidget(id, widget, pose);
}

void temp_viz::Viz3d::removeWidget(const String &id)
{
    impl_->removeWidget(id);
}

temp_viz::Widget temp_viz::Viz3d::getWidget(const String &id) const
{
    return impl_->getWidget(id);
}

void temp_viz::Viz3d::setWidgetPose(const String &id, const Affine3f &pose)
{
    impl_->setWidgetPose(id, pose);
}

void temp_viz::Viz3d::updateWidgetPose(const String &id, const Affine3f &pose)
{
    impl_->updateWidgetPose(id, pose);
}

temp_viz::Affine3f temp_viz::Viz3d::getWidgetPose(const String &id) const
{
    return impl_->getWidgetPose(id);
}

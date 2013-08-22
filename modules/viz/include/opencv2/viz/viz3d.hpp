#pragma once

#if !defined YES_I_AGREE_THAT_VIZ_API_IS_NOT_STABLE_NOW_AND_BINARY_COMPARTIBILITY_WONT_BE_SUPPORTED
    //#error "Viz is in beta state now. Please define macro above to use it"
#endif

#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <boost/concept_check.hpp>

namespace cv
{
    namespace viz
    {
        class CV_EXPORTS Viz3d
        {
        public:
            typedef cv::Ptr<Viz3d> Ptr;
            typedef void (*KeyboardCallback)(const KeyboardEvent&, void*);
            typedef void (*MouseCallback)(const MouseEvent&, void*);

            Viz3d(const String& window_name = String());
            Viz3d(const Viz3d&);
            Viz3d& operator=(const Viz3d&);
            ~Viz3d();

            void setBackgroundColor(const Color& color = Color::black());

            //to refactor
            bool addPolygonMesh(const Mesh3d& mesh, const String& id = "polygon");
            bool updatePolygonMesh(const Mesh3d& mesh, const String& id = "polygon");
            bool addPolylineFromPolygonMesh (const Mesh3d& mesh, const String& id = "polyline");
            bool addPolygon(const Mat& cloud, const Color& color, const String& id = "polygon");


            void showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity());
            void removeWidget(const String &id);
            Widget getWidget(const String &id) const;
            void removeAllWidgets();

            void setWidgetPose(const String &id, const Affine3f &pose);
            void updateWidgetPose(const String &id, const Affine3f &pose);
            Affine3f getWidgetPose(const String &id) const;
            
            void setCamera(const Camera &camera);
            Camera getCamera() const;
            Affine3f getViewerPose();
            void setViewerPose(const Affine3f &pose);
            
            void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);
            void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);
            
            Size getWindowSize() const;
            void setWindowSize(const Size &window_size);
            
            String getWindowName() const;

            void spin();
            void spinOnce(int time = 1, bool force_redraw = false);
            bool wasStopped() const;

            void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);
            void registerMouseCallback(MouseCallback callback, void* cookie = 0);
        private:

            struct VizImpl;
            VizImpl* impl_;
            
            void create(const String &window_name);
            void release();
        };

    } /* namespace viz */
} /* namespace cv */




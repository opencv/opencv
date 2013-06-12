#pragma once

#if !defined YES_I_AGREE_THAT_VIZ_API_IS_NOT_STABLE_NOW_AND_BINARY_COMPARTIBILITY_WONT_BE_SUPPORTED
    //#error "Viz is in beta state now. Please define macro above to use it"
#endif

#include <opencv2/core/cvdef.h>
#include <opencv2/core.hpp>


#include <string>
#include <opencv2/viz/types.hpp>
#include <opencv2/viz/events.hpp>

namespace temp_viz
{
    class CV_EXPORTS Viz3d
    {
    public:

        typedef cv::Ptr<Viz3d> Ptr;

        Viz3d(const String& window_name = String());
        ~Viz3d();

        void setBackgroundColor(const Color& color = Color::black());

        void addCoordinateSystem(double scale, const Affine3f& t, const String &id = "coordinate");

        void showPointCloud(const String& id, InputArray cloud, InputArray colors, const Affine3f& pose = Affine3f::Identity());
        void showPointCloud(const String& id, InputArray cloud, const Color& color, const Affine3f& pose = Affine3f::Identity());

        bool addPointCloudNormals (const Mat &cloud, const Mat& normals, int level = 100, float scale = 0.02f, const String &id = "cloud");

        bool addPlane (const ModelCoefficients &coefficients, const String &id = "plane");
        bool addPlane (const ModelCoefficients &coefficients, double x, double y, double z, const String &id = "plane");
        bool removeCoordinateSystem (const String &id = "coordinate");

        bool addPolygonMesh (const Mesh3d& mesh, const String &id = "polygon");
        bool updatePolygonMesh (const Mesh3d& mesh, const String &id = "polygon");

        bool addPolylineFromPolygonMesh (const Mesh3d& mesh, const String &id = "polyline");


        bool addText (const String &text, int xpos, int ypos, const Color& color, int fontsize = 10, const String &id = "");


        bool addPolygon(const Mat& cloud, const Color& color, const String &id = "polygon");

        bool addSphere (const Point3f &center, double radius, const Color& color, const String &id = "sphere");


        void spin ();
        void spinOnce (int time = 1, bool force_redraw = false);

        void registerKeyboardCallback(void (*callback)(const cv::KeyboardEvent&, void*), void* cookie = 0);
        void registerMouseCallback(void (*callback)(const cv::MouseEvent&, void*), void* cookie = 0);

        bool wasStopped() const;
    private:
        Viz3d(const Viz3d&);
        Viz3d& operator=(const Viz3d&);

        struct VizImpl;
        VizImpl* impl_;
    };
}




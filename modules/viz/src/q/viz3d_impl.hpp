#pragma once

#include <opencv2/core.hpp>
#include <opencv2/viz/events.hpp>
#include <q/interactor_style.h>
#include <q/viz_types.h>
#include <q/common.h>
#include <opencv2/viz/types.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/viz3d.hpp>

namespace temp_viz
{

class CV_EXPORTS Viz3d::VizImpl
{
public:
    typedef cv::Ptr<VizImpl> Ptr;

    VizImpl (const String &name = String());

    virtual ~VizImpl ();
    void setFullScreen (bool mode);
    void setWindowName (const String &name);
    
    /** \brief  Register a callback function for keyboard input
      * \param[in] callback function that will be registered as a callback for a keyboard event
      * \param[in] cookie for passing user data to callback
      */
    void registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void* cookie = 0);
    
    /** \brief Register a callback function for mouse events
          * \param[in] ccallback function that will be registered as a callback for a mouse event
          * \param[in] cookie for passing user data to callback
          */
    void registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie = 0);

    void spin ();
    void spinOnce (int time = 1, bool force_redraw = false);

    bool removePointCloud (const String& id = "cloud");
    inline bool removePolygonMesh (const String& id = "polygon")
    {
        // Polygon Meshes are represented internally as point clouds with special cell array structures since 1.4
        return removePointCloud (id);
    }
    bool removeShape (const String& id = "cloud");

    bool removeText3D (const String& id = "cloud");
    bool removeAllPointClouds ();
    bool removeAllShapes ();

    void setBackgroundColor (const Color& color);

    bool addText (const String &text, int xpos, int ypos, const Color& color, int fontsize = 10, const String& id = "");
    bool updateText (const String &text, int xpos, int ypos, const Color& color, int fontsize = 10, const String& id = "");

    /** \brief Set the pose of an existing shape. Returns false if the shape doesn't exist, true if the pose was succesfully updated. */
    bool updateShapePose (const String& id, const Affine3f& pose);

    bool addText3D (const String &text, const Point3f &position, const Color& color, double textScale = 1.0, const String& id = "");

    bool addPointCloudNormals (const cv::Mat &cloud, const cv::Mat& normals, int level = 100, float scale = 0.02f, const String& id = "cloud");

    /** \brief If the id exists, updates the point cloud; otherwise, adds a new point cloud to the scene
      * \param[in] id a variable to identify the point cloud
      * \param[in] cloud cloud input in x,y,z coordinates
      * \param[in] colors color input in the same order of the points or single uniform color
      * \param[in] pose transform to be applied on the point cloud
      */
    void showPointCloud(const String& id, InputArray cloud, InputArray colors, const Affine3f& pose = Affine3f::Identity());
    void showPointCloud(const String& id, InputArray cloud, const Color& color, const Affine3f& pose = Affine3f::Identity());

    bool addPolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const String& id = "polygon");
    bool updatePolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const String& id = "polygon");

    bool addPolylineFromPolygonMesh (const Mesh3d& mesh, const String& id = "polyline");

    void setPointCloudColor (const Color& color, const String& id = "cloud");
    bool setPointCloudRenderingProperties (int property, double value, const String& id = "cloud");
    bool getPointCloudRenderingProperties (int property, double &value, const String& id = "cloud");

    bool setShapeRenderingProperties (int property, double value, const String& id);
    void setShapeColor (const Color& color, const String& id);

    /** \brief Set whether the point cloud is selected or not
         *  \param[in] selected whether the cloud is selected or not (true = selected)
         *  \param[in] id the point cloud object id (default: cloud)
         */
    bool setPointCloudSelected (const bool selected, const String& id = "cloud" );

    /** \brief Returns true when the user tried to close the window */
    bool wasStopped () const { if (interactor_ != NULL) return (stopped_); else return true; }

    /** \brief Set the stopped flag back to false */
    void resetStoppedFlag () { if (interactor_ != NULL) stopped_ = false; }

    /** \brief Stop the interaction and close the visualizaton window. */
    void close ()
    {
        stopped_ = true;
        // This tends to close the window...
        interactor_->TerminateApp ();
    }
    
    bool addPolygon(const cv::Mat& cloud, const Color& color, const String& id = "polygon");
    bool addArrow (const Point3f& pt1, const Point3f& pt2, const Color& color, bool display_length, const String& id = "arrow");
    bool addArrow (const Point3f& pt1, const Point3f& pt2, const Color& color_line, const Color& color_text, const String& id = "arrow");

    // Add a vtkPolydata as a mesh
    bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, const String& id = "PolyData");
    bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, vtkSmartPointer<vtkTransform> transform, const String& id = "PolyData");
    bool addModelFromPLYFile (const String &filename, const String& id = "PLYModel");
    bool addModelFromPLYFile (const String &filename, vtkSmartPointer<vtkTransform> transform, const String& id = "PLYModel");

    /** \brief Changes the visual representation for all actors to surface representation. */
    void setRepresentationToSurfaceForAllActors ();

    /** \brief Changes the visual representation for all actors to points representation. */
    void setRepresentationToPointsForAllActors ();

    /** \brief Changes the visual representation for all actors to wireframe representation. */
    void setRepresentationToWireframeForAllActors ();

    /** \brief Initialize camera parameters with some default values. */
    void initCameraParameters ();

    /** \brief Search for camera parameters at the command line and set them internally.
        bool getCameraParameters (int argc, char **argv);

        /** \brief Checks whether the camera parameters were manually loaded from file.*/
    bool cameraParamsSet () const;

    /** \brief Update camera parameters and render. */
    void updateCamera ();

    /** \brief Reset camera parameters and render. */
    void resetCamera ();

    /** \brief Reset the camera direction from {0, 0, 0} to the center_{x, y, z} of a given dataset.
          * \param[in] id the point cloud object id (default: cloud)
          */
    void resetCameraViewpoint (const String& id = "cloud");

    /** \brief Set the camera pose given by position, viewpoint and up vector
          * \param[in] pos_x the x coordinate of the camera location
          * \param[in] pos_y the y coordinate of the camera location
          * \param[in] pos_z the z coordinate of the camera location
          * \param[in] view_x the x component of the view point of the camera
          * \param[in] view_y the y component of the view point of the camera
          * \param[in] view_z the z component of the view point of the camera
          * \param[in] up_x the x component of the view up direction of the camera
          * \param[in] up_y the y component of the view up direction of the camera
          * \param[in] up_z the y component of the view up direction of the camera
          */
    void setCameraPosition (const cv::Vec3d& pos, const cv::Vec3d& view, const cv::Vec3d& up);

    /** \brief Set the camera location and viewup according to the given arguments
          * \param[in] pos_x the x coordinate of the camera location
          * \param[in] pos_y the y coordinate of the camera location
          * \param[in] pos_z the z coordinate of the camera location
          * \param[in] up_x the x component of the view up direction of the camera
          * \param[in] up_y the y component of the view up direction of the camera
          * \param[in] up_z the z component of the view up direction of the camera
          */
    void setCameraPosition (double pos_x, double pos_y, double pos_z, double up_x, double up_y, double up_z);

    /** \brief Set the camera parameters via an intrinsics and and extrinsics matrix
          * \note This assumes that the pixels are square and that the center of the image is at the center of the sensor.
          * \param[in] intrinsics the intrinsics that will be used to compute the VTK camera parameters
          * \param[in] extrinsics the extrinsics that will be used to compute the VTK camera parameters
          */
    void setCameraParameters (const cv::Matx33f& intrinsics, const Affine3f& extrinsics);

    /** \brief Set the camera parameters by given a full camera data structure.
          * \param[in] camera camera structure containing all the camera parameters.
          */
    void setCameraParameters (const Camera &camera);

    /** \brief Set the camera clipping distances.
          * \param[in] near the near clipping distance (no objects closer than this to the camera will be drawn)
          * \param[in] far the far clipping distance (no objects further away than this to the camera will be drawn)
          */
    void setCameraClipDistances (double near, double far);

    /** \brief Set the camera vertical field of view in radians */
    void setCameraFieldOfView (double fovy);

    /** \brief Get the current camera parameters. */
    void getCameras (Camera& camera);

    /** \brief Get the current viewing pose. */
    Affine3f getViewerPose ();
    void saveScreenshot (const String &file);

    /** \brief Return a pointer to the underlying VTK Render Window used. */
    //vtkSmartPointer<vtkRenderWindow> getRenderWindow () { return (window_); }

    void setPosition (int x, int y);
    void setSize (int xw, int yw);
    
    void showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity());
    void removeWidget(const String &id);
    Widget getWidget(const String &id) const;
    
    void setWidgetPose(const String &id, const Affine3f &pose);
    void updateWidgetPose(const String &id, const Affine3f &pose);
    Affine3f getWidgetPose(const String &id) const;
    
    void all_data();

private:
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    struct ExitMainLoopTimerCallback : public vtkCommand
    {
        static ExitMainLoopTimerCallback* New()
        {
            return new ExitMainLoopTimerCallback;
        }
        virtual void Execute(vtkObject* vtkNotUsed(caller), unsigned long event_id, void* call_data)
        {
            if (event_id != vtkCommand::TimerEvent)
                return;

            int timer_id = *reinterpret_cast<int*> (call_data);
            if (timer_id != right_timer_id)
                return;

            // Stop vtk loop and send notification to app to wake it up
            viz_->interactor_->TerminateApp ();
        }
        int right_timer_id;
        VizImpl* viz_;
    };

    struct ExitCallback : public vtkCommand
    {
        static ExitCallback* New ()
        {
            return new ExitCallback;
        }
        virtual void Execute (vtkObject*, unsigned long event_id, void*)
        {
            if (event_id == vtkCommand::ExitEvent)
            {
                viz_->stopped_ = true;
                viz_->interactor_->TerminateApp ();
            }
        }
        VizImpl* viz_;
    };

    /** \brief Set to false if the interaction loop is running. */
    bool stopped_;

    double s_lastDone_;

    /** \brief Global timer ID. Used in destructor only. */
    int timer_id_;

    /** \brief Callback object enabling us to leave the main loop, when a timer fires. */
    vtkSmartPointer<ExitMainLoopTimerCallback> exit_main_loop_timer_callback_;
    vtkSmartPointer<ExitCallback> exit_callback_;

    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkRenderWindow> window_;

    /** \brief The render window interactor style. */
    vtkSmartPointer<InteractorStyle> style_;

    /** \brief Internal list with actor pointers and name IDs for point clouds. */
    cv::Ptr<CloudActorMap> cloud_actor_map_;

    /** \brief Internal list with actor pointers and name IDs for shapes. */
    cv::Ptr<ShapeActorMap> shape_actor_map_;
    
    /** \brief Internal list with actor pointers and name IDs for all widget actors */
    cv::Ptr<WidgetActorMap> widget_actor_map_;

    /** \brief Boolean that holds whether or not the camera parameters were manually initialized*/
    bool camera_set_;

    bool removeActorFromRenderer (const vtkSmartPointer<vtkLODActor> &actor);
    bool removeActorFromRenderer (const vtkSmartPointer<vtkActor> &actor);
    bool removeActorFromRenderer (const vtkSmartPointer<vtkProp> &actor);

    //void addActorToRenderer (const vtkSmartPointer<vtkProp> &actor);


    /** \brief Internal method. Creates a vtk actor from a vtk polydata object.
          * \param[in] data the vtk polydata object to create an actor for
          * \param[out] actor the resultant vtk actor object
          * \param[in] use_scalars set scalar properties to the mapper if it exists in the data. Default: true.
          */
    void createActorFromVTKDataSet (const vtkSmartPointer<vtkDataSet> &data, vtkSmartPointer<vtkLODActor> &actor, bool use_scalars = true);

    /** \brief Updates a set of cells (vtkIdTypeArray) if the number of points in a cloud changes
          * \param[out] cells the vtkIdTypeArray object (set of cells) to update
          * \param[out] initcells a previously saved set of cells. If the number of points in the current cloud is
          * higher than the number of cells in \a cells, and initcells contains enough data, then a copy from it
          * will be made instead of regenerating the entire array.
          * \param[in] nr_points the number of points in the new cloud. This dictates how many cells we need to
          * generate
          */
    void updateCells (vtkSmartPointer<vtkIdTypeArray> &cells, vtkSmartPointer<vtkIdTypeArray> &initcells, vtkIdType nr_points);

    void allocVtkPolyData (vtkSmartPointer<vtkAppendPolyData> &polydata);
    void allocVtkPolyData (vtkSmartPointer<vtkPolyData> &polydata);
    void allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata);
};

//void getTransformationMatrix (const Eigen::Vector4f &origin, const Eigen::Quaternionf& orientation, Eigen::Matrix4f &transformation);

//void convertToVtkMatrix (const Eigen::Matrix4f &m, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix);

void convertToVtkMatrix (const cv::Matx44f& m, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix);
void convertToCvMatrix (const vtkSmartPointer<vtkMatrix4x4> &vtk_matrix, cv::Matx44f &m);

vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f &m);
cv::Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix);

/** \brief Convert origin and orientation to vtkMatrix4x4
      * \param[in] origin the point cloud origin
      * \param[in] orientation the point cloud orientation
      * \param[out] vtk_matrix the resultant VTK 4x4 matrix
      */
void convertToVtkMatrix (const Eigen::Vector4f &origin, const Eigen::Quaternion<float> &orientation, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix);
void convertToEigenMatrix (const vtkSmartPointer<vtkMatrix4x4> &vtk_matrix, Eigen::Matrix4f &m);


struct NanFilter
{
    template<typename _Tp, typename _Msk>
    struct Impl
    {
        typedef Vec<_Tp, 3> _Out;

        static _Out* copy(const Mat& source, _Out* output, const Mat& nan_mask)
        {
            CV_Assert(DataDepth<_Tp>::value == source.depth() && source.size() == nan_mask.size());
            CV_Assert(nan_mask.channels() == 3 || nan_mask.channels() == 4);
            CV_DbgAssert(DataDepth<_Msk>::value == nan_mask.depth());

            int s_chs = source.channels();
            int m_chs = nan_mask.channels();

            for(int y = 0; y < source.rows; ++y)
            {
                const _Tp* srow = source.ptr<_Tp>(y);
                const _Msk* mrow = nan_mask.ptr<_Msk>(y);

                for(int x = 0; x < source.cols; ++x, srow += s_chs, mrow += m_chs)
                    if (!isNan(mrow[0]) && !isNan(mrow[1]) && !isNan(mrow[2]))
                        *output++ = _Out(srow);
            }
            return output;
        }
    };

    template<typename _Tp>
    static inline Vec<_Tp, 3>* copy(const Mat& source, Vec<_Tp, 3>* output, const Mat& nan_mask)
    {
        CV_Assert(nan_mask.depth() == CV_32F || nan_mask.depth() == CV_64F);

        typedef Vec<_Tp, 3>* (*copy_func)(const Mat&, Vec<_Tp, 3>*, const Mat&);
        const static copy_func table[2] = { &NanFilter::Impl<_Tp, float>::copy, &NanFilter::Impl<_Tp, double>::copy };

        return table[nan_mask.depth() - 5](source, output, nan_mask);
    }
};

struct ApplyAffine
{
    const Affine3f& affine_;
    ApplyAffine(const Affine3f& affine) : affine_(affine) {}

    template<typename _Tp> Point3_<_Tp> operator()(const Point3_<_Tp>& p) const { return affine_ * p; }

    template<typename _Tp> Vec<_Tp, 3> operator()(const Vec<_Tp, 3>& v) const
    {
        const float* m = affine_.matrix.val;

        Vec<_Tp, 3> result;
        result[0] = (_Tp)(m[0] * v[0] + m[1] * v[1] + m[ 2] * v[2] + m[ 3]);
        result[1] = (_Tp)(m[4] * v[0] + m[5] * v[1] + m[ 6] * v[2] + m[ 7]);
        result[2] = (_Tp)(m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11]);
        return result;
    }
};

}


#pragma once

#include <opencv2/viz.hpp>
#include "interactor_style.h"
#include "viz_types.h"
#include "common.h"

struct cv::viz::Viz3d::VizImpl
{
public:
    typedef cv::Ptr<VizImpl> Ptr;
    typedef Viz3d::KeyboardCallback KeyboardCallback;
    typedef Viz3d::MouseCallback MouseCallback;
    
    int ref_counter;

    VizImpl (const String &name);
    virtual ~VizImpl ();









    //to refactor
    bool removePointCloud (const String& id = "cloud");
    inline bool removePolygonMesh (const String& id = "polygon") { return removePointCloud (id); }
    bool removeShape (const String& id = "cloud");
    bool removeText3D (const String& id = "cloud");
    bool removeAllPointClouds ();

    //create Viz3d::removeAllWidgets()
    bool removeAllShapes ();

    //to refactor
    bool addPolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const String& id = "polygon");
    bool updatePolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const String& id = "polygon");
    bool addPolylineFromPolygonMesh (const Mesh3d& mesh, const String& id = "polyline");

    // to refactor: Widget3D:: & Viz3d::
    bool setPointCloudRenderingProperties (int property, double value, const String& id = "cloud");
    bool getPointCloudRenderingProperties (int property, double &value, const String& id = "cloud");
    bool setShapeRenderingProperties (int property, double value, const String& id);

    /** \brief Set whether the point cloud is selected or not
         *  \param[in] selected whether the cloud is selected or not (true = selected)
         *  \param[in] id the point cloud object id (default: cloud)
         */
    // probably should just remove
    bool setPointCloudSelected (const bool selected, const String& id = "cloud" );








    /** \brief Returns true when the user tried to close the window */
    bool wasStopped () const { if (interactor_ != NULL) return (stopped_); else return true; }

    /** \brief Set the stopped flag back to false */
    void resetStoppedFlag () { if (interactor_ != NULL) stopped_ = false; }

    /** \brief Stop the interaction and close the visualizaton window. */
    void close ()
    {
        stopped_ = true;
        if (interactor_) 
        {
            interactor_->GetRenderWindow()->Finalize();
            interactor_->TerminateApp (); // This tends to close the window...
        }
    }










    
    // to refactor
    bool addPolygon(const cv::Mat& cloud, const Color& color, const String& id = "polygon");
    bool addArrow (const Point3f& pt1, const Point3f& pt2, const Color& color, bool display_length, const String& id = "arrow");
    bool addArrow (const Point3f& pt1, const Point3f& pt2, const Color& color_line, const Color& color_text, const String& id = "arrow");

    // Probably remove this
    bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, const String& id = "PolyData");
    bool addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, vtkSmartPointer<vtkTransform> transform, const String& id = "PolyData");

    // I think this should be moved to 'static Widget Widget::fromPlyFile(const String&)';
    bool addModelFromPLYFile (const String &filename, const String& id = "PLYModel");
    bool addModelFromPLYFile (const String &filename, vtkSmartPointer<vtkTransform> transform, const String& id = "PLYModel");


    // to implement in Viz3d with shorter name
    void setRepresentationToSurfaceForAllActors();
    void setRepresentationToPointsForAllActors();
    void setRepresentationToWireframeForAllActors();








    // ////////////////////////////////////////////////////////////////////////////////////
    // All camera methods to refactor into set/getViewwerPose, setCamera()
    // and 'Camera' class itself with various constructors/fields
    
    void setCamera(const Camera &camera);
    Camera getCamera() const;

    void initCameraParameters (); /** \brief Initialize camera parameters with some default values. */
    bool cameraParamsSet () const; /** \brief Checks whether the camera parameters were manually loaded from file.*/
    void updateCamera (); /** \brief Update camera parameters and render. */
    void resetCamera (); /** \brief Reset camera parameters and render. */

    /** \brief Reset the camera direction from {0, 0, 0} to the center_{x, y, z} of a given dataset.
      * \param[in] id the point cloud object id (default: cloud) */
    void resetCameraViewpoint (const String& id = "cloud");
    
    //to implement Viz3d set/getViewerPose()
    void setViewerPose(const Affine3f &pose);
    Affine3f getViewerPose();

    void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);
    void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);







    //to implemnt in Viz3d
    void saveScreenshot (const String &file);
    void setWindowPosition (int x, int y);
    Size getWindowSize() const;
    void setWindowSize (int xw, int yw);
    void setFullScreen (bool mode);
    void setWindowName (const String &name);
    String getWindowName() const;
    void setBackgroundColor (const Color& color);

    void spin ();
    void spinOnce (int time = 1, bool force_redraw = false);

    void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);
    void registerMouseCallback(MouseCallback callback, void* cookie = 0);








    //declare above (to move to up)
    void showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity());
    void removeWidget(const String &id);
    Widget getWidget(const String &id) const;
    
    void setWidgetPose(const String &id, const Affine3f &pose);
    void updateWidgetPose(const String &id, const Affine3f &pose);
    Affine3f getWidgetPose(const String &id) const; 

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



namespace cv
{
    namespace viz
    {
        //void getTransformationMatrix (const Eigen::Vector4f &origin, const Eigen::Quaternionf& orientation, Eigen::Matrix4f &transformation);
        vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix (const cv::Matx44f &m);
        cv::Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix);

        /** \brief Convert origin and orientation to vtkMatrix4x4
              * \param[in] origin the point cloud origin
              * \param[in] orientation the point cloud orientation
              * \param[out] vtk_matrix the resultant VTK 4x4 matrix
              */
        void convertToVtkMatrix (const Eigen::Vector4f &origin, const Eigen::Quaternion<float> &orientation, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix);

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
                
                static _Out* copyColor(const Mat& source, _Out* output, const Mat& nan_mask)
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
                            {
                                *output = _Out(srow);
                                std::swap((*output)[0], (*output)[2]); // BGR -> RGB
                                ++output;
                            }
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
            
            template<typename _Tp>
            static inline Vec<_Tp, 3>* copyColor(const Mat& source, Vec<_Tp, 3>* output, const Mat& nan_mask)
            {
                CV_Assert(nan_mask.depth() == CV_32F || nan_mask.depth() == CV_64F);

                typedef Vec<_Tp, 3>* (*copy_func)(const Mat&, Vec<_Tp, 3>*, const Mat&);
                const static copy_func table[2] = { &NanFilter::Impl<_Tp, float>::copyColor, &NanFilter::Impl<_Tp, double>::copyColor };

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

        private:
            ApplyAffine(const ApplyAffine&);
            ApplyAffine& operator=(const ApplyAffine&);
        };


        inline Color vtkcolor(const Color& color)
        {
            Color scaled_color = color * (1.0/255.0);
            std::swap(scaled_color[0], scaled_color[2]);
            return scaled_color;
        }

        inline Vec3d vtkpoint(const Point3f& point) { return Vec3d(point.x, point.y, point.z); }
        template<typename _Tp> inline _Tp normalized(const _Tp& v) { return v * 1/cv::norm(v); }
        
        struct ConvertToVtkImage
        {
            struct Impl
            {
                static void copyImageMultiChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
                {
                    int i_chs = image.channels();
            
                    for (int i = 0; i < image.rows; ++i)
                    {
                        const unsigned char * irows = image.ptr<unsigned char>(i);
                        for (int j = 0; j < image.cols; ++j, irows += i_chs)
                        {
                            unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                            memcpy(vrows, irows, i_chs);
                            std::swap(vrows[0], vrows[2]); // BGR -> RGB
                        }
                    }
                    output->Modified();
                }
                
                static void copyImageSingleChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
                {
                    for (int i = 0; i < image.rows; ++i)
                    {
                        const unsigned char * irows = image.ptr<unsigned char>(i);
                        for (int j = 0; j < image.cols; ++j, ++irows)
                        {
                            unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                            *vrows = *irows;
                        }
                    }
                    output->Modified();
                }
            };
            
            static void convert(const Mat &image, vtkSmartPointer<vtkImageData> output)
            {
                // Create the vtk image
                output->SetDimensions(image.cols, image.rows, 1);
                output->SetNumberOfScalarComponents(image.channels());
                output->SetScalarTypeToUnsignedChar();
                output->AllocateScalars();
                
                int i_chs = image.channels();
                if (i_chs > 1)
                {
                    // Multi channel images are handled differently because of BGR <-> RGB
                    Impl::copyImageMultiChannel(image, output);
                }
                else
                {
                    Impl::copyImageSingleChannel(image, output);
                }
            }
        };
    }

}


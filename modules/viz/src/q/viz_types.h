#pragma once

#include "precomp.hpp"

namespace temp_viz
{
    struct CV_EXPORTS CloudActor
    {
        /** \brief The actor holding the data to render. */
        vtkSmartPointer<vtkLODActor> actor;

        /** \brief The viewpoint transformation matrix. */
        vtkSmartPointer<vtkMatrix4x4> viewpoint_transformation_;

        /** \brief Internal cell array. Used for optimizing updatePointCloud. */
        vtkSmartPointer<vtkIdTypeArray> cells;
    };

    typedef std::map<std::string, CloudActor> CloudActorMap;
    typedef std::map<std::string, vtkSmartPointer<vtkProp> > ShapeActorMap;
}


// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_ML_INL_HPP
#define OPENCV_ML_INL_HPP

namespace cv { namespace ml {

// declared in ml.hpp
template<class SimulatedAnnealingSolverSystem>
int simulatedAnnealingSolver(SimulatedAnnealingSolverSystem& solverSystem,
     double initialTemperature, double finalTemperature, double coolingRatio,
     size_t iterationsPerStep,
     CV_OUT double* lastTemperature,
     cv::RNG& rngEnergy
)
{
    CV_Assert(finalTemperature > 0);
    CV_Assert(initialTemperature > finalTemperature);
    CV_Assert(iterationsPerStep > 0);
    CV_Assert(coolingRatio < 1.0f);
    double Ti = initialTemperature;
    double previousEnergy = solverSystem.energy();
    int exchange = 0;
    while (Ti > finalTemperature)
    {
        for (size_t i = 0; i < iterationsPerStep; i++)
        {
            solverSystem.changeState();
            double newEnergy = solverSystem.energy();
            if (newEnergy < previousEnergy)
            {
                previousEnergy = newEnergy;
                exchange++;
            }
            else
            {
                double r = rngEnergy.uniform(0.0, 1.0);
                if (r < std::exp(-(newEnergy - previousEnergy) / Ti))
                {
                    previousEnergy = newEnergy;
                    exchange++;
                }
                else
                {
                    solverSystem.reverseState();
                }
            }
        }
        Ti *= coolingRatio;
    }
    if (lastTemperature)
        *lastTemperature = Ti;
    return exchange;
}

}} //namespace

#endif // OPENCV_ML_INL_HPP

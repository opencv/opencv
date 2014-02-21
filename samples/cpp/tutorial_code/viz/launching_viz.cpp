/**
 * @file launching_viz.cpp
 * @brief Launching visualization window
 * @author Ozan Cagri Tonkal
 */

#include <opencv2/viz/vizcore.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function help
 * @brief Display instructions to use this tutorial program
 */
void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to launch a 3D visualization window. You can stop event loop to continue executing. "
    << "You can access the same window via its name. You can run event loop for a given period of time. " << endl
    << "Usage:"                                                                     << endl
    << "./launching_viz"                                                            << endl
    << endl;
}

/**
 * @function main
 */
int main()
{
    help();
    /// Create a window
    viz::Viz3d myWindow("Viz Demo");

    /// Start event loop
    myWindow.spin();

    /// Event loop is over when pressed q, Q, e, E
    cout << "First event loop is over" << endl;

    /// Access window via its name
    viz::Viz3d sameWindow = viz::getWindowByName("Viz Demo");

    /// Start event loop
    sameWindow.spin();

    /// Event loop is over when pressed q, Q, e, E
    cout << "Second event loop is over" << endl;

    /// Event loop is over when pressed q, Q, e, E
    /// Start event loop once for 1 millisecond
    sameWindow.spinOnce(1, true);
    while(!sameWindow.wasStopped())
    {
        /// Interact with window

        /// Event loop for 1 millisecond
        sameWindow.spinOnce(1, true);
    }

    /// Once more event loop is stopped
    cout << "Last event loop is over" << endl;
    return 0;
}

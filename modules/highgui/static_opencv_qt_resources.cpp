/**
 * Part of the Hunter patch for static linking OpenCV with Qt
 * Initializes the resources for the highgui window icons
 */

#include <QtCore>

inline void static_opencv_qt_resources() {
	Q_INIT_RESOURCE(window_QT);
}

namespace {

    struct static_opencv_qt_resources_initializer{
        static_opencv_qt_resources_initializer() {
            static_opencv_qt_resources();
        }
    };
    
    static_opencv_qt_resources_initializer global_instance;
}


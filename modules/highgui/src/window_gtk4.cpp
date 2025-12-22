#ifdef HAVE_GTK4

#include <gtk/gtk.h>
#include "precomp.hpp"

namespace cv {
namespace highgui {

bool GTK4BackendAvailable()
{
    return true;
}

} // namespace highgui
} // namespace cv

#endif

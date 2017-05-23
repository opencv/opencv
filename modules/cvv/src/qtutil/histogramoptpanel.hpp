#ifndef CVVISUAL_HISTOGRAM_OPT_PANEL
#define CVVISUAL_HISTOGRAM_OPT_PANEL

#include <QWidget>

#include "histogram.hpp"

namespace cvv
{
namespace qtutil
{

class HistogramOptPanel
  : public QWidget
{
  Q_OBJECT

  public:
    HistogramOptPanel(const Histogram& hist, bool showHideButton = true, QWidget* parent = nullptr);

};

}
}

#endif // CVVISUAL_HISTOGRAM_OPT_PANEL

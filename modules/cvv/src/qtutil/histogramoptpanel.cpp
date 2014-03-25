#include "histogramoptpanel.hpp"

#include <QVBoxLayout>
#include <QCheckBox>


namespace cvv
{
namespace qtutil
{

HistogramOptPanel::HistogramOptPanel(const Histogram& hist, bool showHideButton, QWidget* parent)
  :QWidget{parent}
{
  auto layout = new QVBoxLayout();
  layout->setContentsMargins(0,0,0,0);

  if (showHideButton) {
    auto showCheckbox = new QCheckBox("Show Histogram");
    showCheckbox->setChecked(false);
    connect(showCheckbox, SIGNAL(clicked(bool)), &hist, SLOT(setVisible(bool)));
    layout->addWidget(showCheckbox);
  }

  setLayout(layout);
}

}
}

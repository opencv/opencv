#ifndef CVVISUAL_IMAGE_CALL_TAB_HPP
#define CVVISUAL_IMAGE_CALL_TAB_HPP

#include <QHBoxLayout>
#include <QString>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include "call_tab.hpp"
#include "../view/image_view.hpp"
#include "../controller/view_controller.hpp"
#include "../impl/single_image_call.hpp"

namespace cvv
{
namespace gui
{

/** Single Image Call Tab.
 * @brief Inner part of a tab, contains an IageView.
 * The inner part of a tab or window
 * containing an ImageView.
 * Allows to access the help.
 */
class ImageCallTab : public CallTab
{
	Q_OBJECT

      public:
	/**
	 * @brief Short constructor named after the Call.
	 * Initializes the ImageCallTab and names it after the associated
	 * FilterCall.
	 * @param call the SingleImageCall containing the information to be
	 * visualized.
	 */
	ImageCallTab(const cvv::impl::SingleImageCall &call);

	/**
	 * @brief Constructor using default view.
	 * Short constructor..
	 * @param tabName.
	 * @param call the SingleImageCall containing the information to be
	 * visualized.
	 * @attention might be deleted.
	 */
	ImageCallTab(const QString &tabName,
	             const cvv::impl::SingleImageCall &call);

	/**
	 * @brief get ID.
	 * @return the ID of the CallTab.
	 * (ID is equal to the ID of the associated call).
	 * Overrides CallTab's getId.
	 */
	size_t getId() const override;

      private
slots:

	/**
	 * @brief Help Button clicked.
	 * Called when the help button is clicked.
	 */
	void helpButtonClicked() const;

      private:
	/**
	 * @brief Sets up the visible parts.
	 * Called by the constructors.
	 */
	void createGui();

	/**
	 * @brief sets up View referred to by viewId.
	 * @param viewId ID of the view to be set.
	 * @throw std::out_of_range if no view named viewId was registered.
	 */
	void setView();

	util::Reference<const cvv::impl::SingleImageCall> imageCall_;
	cvv::view::ImageView *imageView_;

	QPushButton *helpButton_;
	QHBoxLayout *hlayout_;
	QVBoxLayout *vlayout_;
	QWidget *upperBar_;
};
}
} // namespaces

#endif

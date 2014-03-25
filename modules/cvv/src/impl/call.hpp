#ifndef CVVISUAL_CALL_HPP
#define CVVISUAL_CALL_HPP

#include <utility>

#include <QString>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "call_meta_data.hpp"


namespace cvv
{
namespace impl
{

/**
 * @brief Returns a new, unique id for calls.
 */
size_t newCallId();

/**
 * @brief Baseclass for all calls. Provides access to the common functionality.
 */
class Call
{
      public:
	virtual ~Call()
	{
	}

	/**
	 * @brief Returns the unique id of the call.
	 */
	size_t getId() const
	{
		return id;
	}

	/**
	 * Returns a string that identifies what kind of call this is.
	 */
	const QString &type() const
	{
		return calltype;
	}

	/**
	 * @brief Returns the number of images that are part of the call.
	 */
	virtual size_t matrixCount() const = 0;

	/**
	 * Returns the n'th matrix that is involved in the call.
	 *
	 * @throws std::out_of_range if index is higher then matrixCount().
	 */
	virtual const cv::Mat &matrixAt(size_t index) const = 0;

	/**
	 * @brief provides a description of the call.
	 */
	const QString &description() const
	{
		return description_;
	}

	/**
	 * @brief Returns a string which view was requested by the caller of the
	 * API.
	 */
	const QString &requestedView() const
	{
		return requestedView_;
	}

	/**
	 * @brief Provides read-access to the meta-data of the call (from where
	 * it came).
	 */
	const CallMetaData &metaData() const
	{
		return metaData_;
	}

      protected:
	/**
	 * @brief Default-construcs a new Call with a new, unique ID.
	 */
	Call();

	/**
	 * @brief Construcs a new Call with a new, unique ID and the provided
	 * data.
	 */
	Call(impl::CallMetaData callData, QString type, QString description,
	     QString requestedView);

	Call(const Call &) = default;
	Call(Call &&) = default;

	Call &operator=(const Call &) = default;
	Call &operator=(Call &&) = default;

	impl::CallMetaData metaData_;
	size_t id;
	QString calltype;
	QString description_;
	QString requestedView_;
};
}
} // namespaces

#endif

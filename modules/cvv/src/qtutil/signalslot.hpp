#ifndef CVVISUAL_SIGNALEMITTER_HPP
#define CVVISUAL_SIGNALEMITTER_HPP

// std
#include <functional>
#include <stdexcept>

#include "opencv2/core/core.hpp"

// QT
#include <QObject>
#include <QString>


namespace cvv
{
namespace qtutil
{

/**
 * @brief The Signal class can be used for privat or static signals and in case
 * of
 * conflicts between templates and Q_OBJECT.
 */
class Signal : public QObject
{
	Q_OBJECT
      public:
	/**
	 * @brief Constructor
	 * @param parent The parent
	 */
	Signal(QObject *parent = nullptr) : QObject{ parent }
	{
	}

	/**
	 * @brief Emits the signal.
	 * @param args The arguments
	 */
	void emitSignal() const
	{
		emit signal();
	}
signals:
	/**
	 * @brief The signal emited by emitSignal.
	 */
	void signal() const;
};

/**
 * @brief The Slot class can be used for static slots and in case of conflicts
 * between
 * templates and Q_OBJECT.
 */
class Slot : public QObject
{
	Q_OBJECT
      public:
	/**
	 * @brief Constructor
	 * @param f Function called by the slot slot()
	 * @throw std::invalid_argument If f is invalide
	 * @param parent The parent
	 */
	Slot(const std::function<void()> &f, QObject *parent = nullptr)
	    : QObject{ parent }, function_{ f }
	{
		if (!f)
			throw std::invalid_argument{ "invalid function" };
	}

      public
slots:
	/**
	 * @brief The slot calling function()
	 */
	void slot() const
	{
		function_();
	}

      private:
	/**
	 * @brief The function called by the slot slot()
	 */
	std::function<void()> function_;
};

// ///////////////////////////////////////////////////////////////
// manual "templating" for classes Signal and Slot
// ///////////////////////////////////////////////////////////////

/**
 * @brief Similar to Signal (difference: it accepts a QStriang).
 */
class SignalQString : public QObject
{
	Q_OBJECT
      public:
	SignalQString(QObject *parent = nullptr) : QObject{ parent }
	{
	}

	void emitSignal(const QString &t) const
	{
		emit signal(t);
	}
signals:
	void signal(QString t) const;
};

/**
 * @brief Similar to Slot (difference: it accepts a QString).
 */
class SlotQString : public QObject
{
	Q_OBJECT
      public:
	SlotQString(const std::function<void(QString)> &f,
	            QObject *parent = nullptr)
	    : QObject{ parent }, function_{ f }
	{
		if (!f)
			throw std::invalid_argument{ "invalide function" };
	}

	~SlotQString()
	{
	}

      public
slots:
	void slot(QString t) const
	{
		function_(t);
	}

      private:
	std::function<void(QString)> function_;
};

/**
 * @brief Similar to Signal (difference: it accepts a cv::Mat&).
 */
class SignalMatRef : public QObject
{
	Q_OBJECT
      public:
	SignalMatRef(QObject *parent = nullptr) : QObject{ parent }
	{
	}

	void emitSignal(cv::Mat &mat) const
	{
		emit signal(mat);
	}
signals:
	/**
	 * @brief The signal emited by emitSignal.
	 */
	void signal(cv::Mat &mat) const;
};

/**
 * @brief Similar to Slot (difference: it accepts a bool).
 */
class SlotBool : public QObject
{
	Q_OBJECT
      public:
	SlotBool(const std::function<void(bool)> &f, QObject *parent = nullptr)
	    : QObject{ parent }, function_{ f }
	{
		if (!f)
			throw std::invalid_argument{ "invalide function" };
	}

      public
slots:
	void slot(bool t) const
	{
		function_(t);
	}

      private:
	std::function<void(bool)> function_;
};
}
} // end namespaces qtutil, cvv
#endif // CVVISUAL_SIGNALEMITTER_HPP

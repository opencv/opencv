#include "util.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <functional>

#include <opencv/highgui.h>

#include <QDesktopServices>
#include <QUrl>
#include <QSettings>

#include "types.hpp"

namespace cvv
{
namespace qtutil
{

QSet<QString> createStringSet(QString string)
{
	QSet<QString> set;
	set.insert(string);
	return set;
}

std::pair<bool, QString> typeToQString(const cv::Mat &mat)
{
	QString s{};
	bool b = true;
	switch (mat.depth())
	{
	case CV_8U:
		s.append("CV_8U");
		break;
	case CV_8S:
		s.append("CV_8S");
		break;
	case CV_16U:
		s.append("CV_16U");
		break;
	case CV_16S:
		s.append("CV_16S");
		break;
	case CV_32S:
		s.append("CV_32S");
		break;
	case CV_32F:
		s.append("CV_32F");
		break;
	case CV_64F:
		s.append("CV_64F");
		break;
	default:
		s.append("DEPTH<").append(QString::number(mat.depth())).append(">");
		b = false;
	}
	s.append("C").append(QString::number(mat.channels()));
	return { b, s };
}

QString conversionResultToString(const ImageConversionResult &result)
{
	switch (result)
	{
	case ImageConversionResult::SUCCESS:
		return "SUCCESS";
		break;
	case ImageConversionResult::MAT_EMPTY:
		return "Empty mat";
		break;
	case ImageConversionResult::MAT_NOT_2D:
		return "Mat not two dimensional";
		break;
	case ImageConversionResult::FLOAT_OUT_OF_0_TO_1:
		return "Float values out of range [0,1]";
		break;
	case ImageConversionResult::NUMBER_OF_CHANNELS_NOT_SUPPORTED:
		return "Unsupported number of channels";
		break;
	case ImageConversionResult::MAT_INVALID_SIZE:
		return "Invalid size";
		break;
	case ImageConversionResult::MAT_UNSUPPORTED_DEPTH:
		return "Unsupported depth ";
		break;
	}
	return "Unknown result from convert function";
}

// ////////////////////////////////////////////////////////////////////////////////////////////////
// image conversion stuff
// ////////////////////////////////////////////////
// convert an image with known depth and channels (the number of chanels is the
// suffix (convertX)
namespace structures
{
/**
 * @brief Gray color table for CV_XXC1
 */
struct GrayColorTable
{
	/**
	 * @brief Constructor
	 */
	GrayColorTable() : table{}
	{
		for (int i = 0; i < 256; i++)
		{
			table.push_back(qRgb(i, i, i));
		}
	}

	/**
	 * @brief Destructor
	 */
	~GrayColorTable()
	{
	}

	/**
	 * @brief The color table
	 */
	QVector<QRgb> table;
};

/**
 * @brief Static GrayColorTable for CV_XXC1
 */
const static GrayColorTable grayColorTable{};

// helper
/**
 * @brief Provides the parts of the conversion fuction that differ depending on
 * the type.
 */
template <int depth, int channels> struct ConvertHelper
{
	static_assert(channels >= 1 && channels <= 4,
		      "Illegal number of channels");
	QImage image(const cv::Mat &mat);
	void pixelOperation(int i, int j, const cv::Mat &mat, uchar *row);
};

/**
 * @brief Provides the parts of the conversion fuction that differ depending on
 * the type.
 */
template <int depth> struct ConvertHelper<depth, 1>
{
	static QImage image(const cv::Mat &mat)
	{
		QImage img{ mat.cols, mat.rows, QImage::Format_Indexed8 };
		img.setColorTable(grayColorTable.table);
		return img;
	}

	static void pixelOperation(int i, int j, const cv::Mat &mat, uchar *row)
	{
		row[j] =
		    convertTo8U<depth>(mat.at<PixelType<depth, 1>>(i, j)[0]);
	}
};

/**
 * @brief Provides the parts of the conversion fuction that differ depending on
 * the type.
 */
template <int depth> struct ConvertHelper<depth, 2>
{
	static QImage image(const cv::Mat &mat)
	{
		return QImage{ mat.cols, mat.rows, QImage::Format_RGB888 };
	}

	static void pixelOperation(int i, int j, const cv::Mat &mat, uchar *row)
	{
		row[j * 3] = 0; // r
		row[j * 3 + 1] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 2>>(i, j)[1]); // g
		row[j * 3 + 2] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 2>>(i, j)[0]); // b
	}
};

/**
 * @brief Provides the parts of the conversion fuction that differ depending on
 * the type.
 */
template <int depth> struct ConvertHelper<depth, 3>
{
	static QImage image(const cv::Mat &mat)
	{
		return QImage{ mat.cols, mat.rows, QImage::Format_RGB888 };
	}

	static void pixelOperation(int i, int j, const cv::Mat &mat, uchar *row)
	{
		row[3 * j] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 3>>(i, j)[2]); // r
		row[3 * j + 1] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 3>>(i, j)[1]); // g
		row[3 * j + 2] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 3>>(i, j)[0]); // b
	}
};

/**
 * @brief Provides the parts of the conversion fuction that differ depending on
 * the type.
 */
template <int depth> struct ConvertHelper<depth, 4>
{
	static QImage image(const cv::Mat &mat)
	{
		return QImage{ mat.cols, mat.rows, QImage::Format_ARGB32 };
	}

	static void pixelOperation(int i, int j, const cv::Mat &mat, uchar *row)
	{
		row[4 * j + 3] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 4>>(i, j)[3]); // a
		row[4 * j + 2] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 4>>(i, j)[2]); // r
		row[4 * j + 1] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 4>>(i, j)[1]); // g
		row[4 * j] = convertTo8U<depth>(
		    mat.at<PixelType<depth, 4>>(i, j)[0]); // b
	}
};
}

/**
 * @brief Converts parts of a cv Mat. [minRow,maxRow)
 * @param mat The mat.
 * @param img The result image.
 * @param minRow Row to start.
 * @param maxRow Last row.
 */
template <int depth, int channels>
void convertPart(const cv::Mat &mat, QImage &img, int minRow, int maxRow)
{
	if (minRow == maxRow)
	{
		return;
	}
	if (maxRow < minRow)
	{
		throw std::invalid_argument{ "maxRow<minRow" };
	}
	if (maxRow > mat.rows)
	{
		throw std::invalid_argument{ "maxRow>mat.rows" };
	}
	uchar *row;
	for (int i = minRow; i < maxRow; i++)
	{
		row = img.scanLine(i);
		for (int j = 0; j < mat.cols; j++)
		{
			structures::ConvertHelper<
			    depth, channels>::pixelOperation(i, j, mat, row);
		}
	}
}

/**
 * @brief Converts a cv Mat.
 * @param mat The mat.
 * @param threads The number of threads to use.
 * @return The converted QImage.
 */
template <int depth, int channels>
QImage convert(const cv::Mat &mat, unsigned int threads)
{
	QImage img = structures::ConvertHelper<depth, channels>::image(mat);

	if (threads > 1)
	{
		// multithreadding
		auto nThreads =
		    std::min(threads, std::thread::hardware_concurrency());
		std::vector<std::thread> workerThreads;
		workerThreads.reserve(nThreads);
		int nperthread = mat.rows / nThreads;
		for (std::size_t i = 0; i < nThreads; i++)
		{
			workerThreads.emplace_back(
			    convertPart<depth, channels>, mat, std::ref(img),
			    i * nperthread, i * nperthread + nperthread);
		}
		// there may be some rows left
		convertPart<depth, channels>(mat, img, nperthread * nThreads,
					     mat.rows);

		// join
		for (auto &t : workerThreads)
		{
			t.join();
		}
	}
	else
	{
		convertPart<depth, channels>(mat, img, 0, mat.rows);
	}
	return img;
}

// ////////////////////////////////////////////////
/**
 * @brief Checks wheather all channels of each pixel are in the given range.
 * @param mat The Mat.
 * @param min Minimal value
 * @param max Maximal value
 * @return Wheather all channels of each pixel are in the given range.
 */
template <int depth>
bool checkValueRange(const cv::Mat &mat, DepthType<depth> min,
		     DepthType<depth> max)
{
	std::pair<cv::MatConstIterator_<DepthType<depth>>,
		  cv::MatConstIterator_<DepthType<depth>>>
	mm{ std::minmax_element(mat.begin<DepthType<depth>>(),
				mat.end<DepthType<depth>>()) };

	return cv::saturate_cast<DepthType<CV_8UC1>>(*(mm.first)) >= min &&
	       cv::saturate_cast<DepthType<CV_8UC1>>(*(mm.second)) <= max;
}
// ////////////////////////////////////////////////
// error result
// the error could be printed on an image
// second parameter: maybe more informations are useful
/**
 * @brief Creates the error result for a given error.
 * @param res The error code.
 * @return The result.
 */
std::pair<ImageConversionResult, QImage> errorResult(ImageConversionResult res,
						     const cv::Mat &mat)
{
	switch (res)
	{
	case ImageConversionResult::FLOAT_OUT_OF_0_TO_1:
	case ImageConversionResult::MAT_NOT_2D:
	case ImageConversionResult::MAT_UNSUPPORTED_DEPTH:
	case ImageConversionResult::NUMBER_OF_CHANNELS_NOT_SUPPORTED:
	{
		QImage imgresult{ mat.cols, mat.rows, QImage::Format_RGB444 };
		imgresult.fill(Qt::black);
		return { res, imgresult };
	}
	break;
	case ImageConversionResult::SUCCESS:
		;
	case ImageConversionResult::MAT_EMPTY:
		;
	case ImageConversionResult::MAT_INVALID_SIZE:
		;
	}
	return { res, QImage{ 0, 0, QImage::Format_Invalid } };
}

// split depth
/**
 * @brief Converts a given image. (this step splits according to the depth)
 * @param mat The Mat.
 * @param skipFloatRangeTest Wheather a rangecheck for float images will be
 * performed.
 * @param threads The number of threads to use.
 * @return The converted QImage.
 */
template <int channels>
std::pair<ImageConversionResult, QImage>
convert(const cv::Mat &mat, bool skipFloatRangeTest, unsigned int threads)
{
	// depth ok?
	switch (mat.depth())
	{
	case CV_8U:
		return { ImageConversionResult::SUCCESS,
			 convert<CV_8U, channels>(mat, threads) };
		break;
	case CV_8S:
		return { ImageConversionResult::SUCCESS,
			 convert<CV_8S, channels>(mat, threads) };
		break;
	case CV_16U:
		return { ImageConversionResult::SUCCESS,
			 convert<CV_16U, channels>(mat, threads) };
		break;
	case CV_16S:
		return { ImageConversionResult::SUCCESS,
			 convert<CV_16S, channels>(mat, threads) };
		break;
	case CV_32S:
		return { ImageConversionResult::SUCCESS,
			 convert<CV_32S, channels>(mat, threads) };
		break;
	case CV_32F:
		if (!skipFloatRangeTest)
		{
			if (!checkValueRange<CV_32F>(
				 mat, cv::saturate_cast<DepthType<CV_32F>>(0),
				 cv::saturate_cast<DepthType<CV_32F>>(
				     1))) // floating depth + in range [0,1]
			{
				return errorResult(
				    ImageConversionResult::FLOAT_OUT_OF_0_TO_1,
				    mat);
			}
		}
		return { ImageConversionResult::SUCCESS,
			 convert<CV_32F, channels>(mat, threads) };
		break;
	case CV_64F:
		if (!skipFloatRangeTest)
		{
			if (!checkValueRange<CV_64F>(
				 mat, cv::saturate_cast<DepthType<CV_64F>>(0),
				 cv::saturate_cast<DepthType<CV_64F>>(
				     1))) // floating depth + in range [0,1]
			{
				return errorResult(
				    ImageConversionResult::FLOAT_OUT_OF_0_TO_1,
				    mat);
			}
		}
		return { ImageConversionResult::SUCCESS,
			 convert<CV_64F, channels>(mat, threads) };
		break;
	default:
		return errorResult(ImageConversionResult::MAT_UNSUPPORTED_DEPTH,
				   mat);
	}
}

// convert
/*
 * @brief Converts a given image. (this step splits according to the channels)
 * @param mat The Mat.
 * @param skipFloatRangeTest Wheather a rangecheck for float images will be
 * performed.
 * @param threads The number of threads to use.
 * @return The converted QImage.
 */
std::pair<ImageConversionResult, QImage>
convertMatToQImage(const cv::Mat &mat, bool skipFloatRangeTest,
		   unsigned int threads)
{
	// empty?
	if (mat.empty())
	{
		return errorResult(ImageConversionResult::MAT_EMPTY, mat);
	};

	// 2d?
	if (mat.dims != 2)
	{
		return errorResult(ImageConversionResult::MAT_NOT_2D, mat);
	};

	// size ok
	if (mat.rows < 1 || mat.cols < 1)
	{
		return errorResult(ImageConversionResult::MAT_INVALID_SIZE,
				   mat);
	}

	// check channels 1-4
	// now convert
	switch (mat.channels())
	{
	case 1:
		return convert<1>(mat, skipFloatRangeTest, threads);
		break;
	case 2:
		return convert<2>(mat, skipFloatRangeTest, threads);
		break;
	case 3:
		return convert<3>(mat, skipFloatRangeTest, threads);
		break;
	case 4:
		return convert<4>(mat, skipFloatRangeTest, threads);
		break;
	default:
		return errorResult(
		    ImageConversionResult::NUMBER_OF_CHANNELS_NOT_SUPPORTED,
		    mat);
	}
	// floating depth + in range [0,1]  (in function convert<T>)
	// depth ok?						(in function
	// convert<T>)
}

std::pair<ImageConversionResult, QPixmap>
convertMatToQPixmap(const cv::Mat &mat, bool skipFloatRangeTest,
		    unsigned int threads)
{
	auto converted = convertMatToQImage(mat, skipFloatRangeTest, threads);
	return { converted.first, QPixmap::fromImage(converted.second) };
}

std::vector<cv::Mat> splitChannels(const cv::Mat &mat)
{
	if (mat.channels() < 1)
	{
		return std::vector<cv::Mat>{};
	}
	auto chan = std::unique_ptr<cv::Mat[]>(new cv::Mat[mat.channels()]);
	cv::split(mat, chan.get());
	std::vector<cv::Mat> result{};
	// put in vector
	for (int i = 0; i < mat.channels(); i++)
	{
		result.emplace_back(chan[i]);
	}
	return result;
}

cv::Mat mergeChannels(std::vector<cv::Mat> mats)
{
	if (mats.size() <= 0)
	{
		throw std::invalid_argument{ "no input mat" };
	}

	// check
	if (mats.at(0).channels() != 1)
	{
		throw std::invalid_argument{ "mat 0 not 1 channel" };
	}
	int type = mats.at(0).type();
	auto size = mats.at(0).size();
	for (std::size_t i = 1; i < mats.size(); i++)
	{
		if ((type != mats.at(i).type()) || (size != mats.at(i).size()))
		{
			throw std::invalid_argument{
				"mats have different sizes or depths."
				"(or not 1 channel)"
			};
		}
	}
	// merge
	cv::Mat result{ mats.at(0).rows, mats.at(0).cols, mats.at(0).type() };

	std::unique_ptr<cv::Mat[]> mergeinput(new cv::Mat[mats.size()]);
	for (std::size_t i = 0; i < mats.size(); i++)
	{
		mergeinput[i] = mats.at(i);
	}
	merge(mergeinput.get(), mats.size(), result);

	return result;
}

void openHelpBrowser(const QString &topic)
{
	auto topicEncoded = QUrl::toPercentEncoding(topic);
	QDesktopServices::openUrl(
	    QUrl(QString("http://cvv.mostlynerdless.de/help.php?topic=") +
		 topicEncoded));
}

void setDefaultSetting(const QString &scope, const QString &key,
		       const QString &value)
{
	QSettings settings{ "CVVisual", QSettings::IniFormat };
	QString _key = scope + "/" + key;
	if (!settings.contains(_key))
	{
		settings.setValue(_key, value);
	}
}

void setSetting(const QString &scope, const QString &key, const QString &value)
{
	QSettings settings{ "CVVisual", QSettings::IniFormat };
	QString _key = scope + "/" + key;
	settings.setValue(_key, value);
}

QString getSetting(const QString &scope, const QString &key)
{
	QSettings settings{ "CVVisual", QSettings::IniFormat };
	QString _key = scope + "/" + key;
	if (!settings.contains(_key))
	{
		throw std::invalid_argument{ "there is no such setting" };
	}
	QString set = settings.value(_key).value<QString>();
	return set;
}
}
}

#include "rawview_table_row.hpp"

#include <functional>
#include <iostream>
#include <utility>

#include <QTableWidgetItem>
#include <QImage>

#include "../qtutil/util.hpp"
#include "../stfl/stringutils.hpp"

namespace cvv
{
namespace gui
{

RawviewTableRow::RawviewTableRow(cv::DMatch match, cv::KeyPoint keyPoint1,
                                 cv::KeyPoint keyPoint2)
    : match{ match }, keyPoint1{ keyPoint1 }, keyPoint2{ keyPoint2 },
      hasLonelyKeyPoint_{ false }
{
}

RawviewTableRow::RawviewTableRow(cv::KeyPoint keyPoint, bool left)
    : keyPoint1{ keyPoint }, hasLonelyKeyPoint_{ true }, left{ left }
{
}

void RawviewTableRow::addToTable(QTableWidget *table, size_t row)
{
	std::vector<QString> items;
	if (!hasLonelyKeyPoint_)
	{
		items = { QString::number(matchDistance()),
			  QString::number(matchImgIdx()),
			  QString::number(matchQueryIdx()),
			  QString::number(matchTrainIdx()),
			  QString::number(keyPoint1XCoord()),
			  QString::number(keyPoint1YCoord()),
			  QString::number(keyPoint1Size()),
			  QString::number(keyPoint1Angle()),
			  QString::number(keyPoint1Response()),
			  QString::number(keyPoint1Octave()),
			  QString::number(keyPoint1ClassId()),
			  QString::number(keyPoint2XCoord()),
			  QString::number(keyPoint2YCoord()),
			  QString::number(keyPoint2Size()),
			  QString::number(keyPoint2Angle()),
			  QString::number(keyPoint2Response()),
			  QString::number(keyPoint2Octave()),
			  QString::number(keyPoint2ClassId()) };
	}
	else
	{
		items = { QString::number(keyPoint1XCoord()),
			  QString::number(keyPoint1YCoord()),
			  QString::number(keyPoint1Size()),
			  QString::number(keyPoint1Angle()),
			  QString::number(keyPoint1Response()),
			  QString::number(keyPoint1Octave()),
			  QString::number(keyPoint1ClassId()),
			  QString::number(left ? 1 : 2) };
	}
	for (size_t i = 0; i < items.size(); i++)
	{
		auto *item = new QTableWidgetItem(items[i]);
		item->setFlags(item->flags() ^ Qt::ItemIsEditable);
		table->setItem(row, i, item);
	}
}

QString RawviewTableRow::rowsToText(const std::vector<RawviewTableRow> &rows,
                                    const QString format,
                                    bool singleKeyPointRows)
{
	QStringList lines;
	// header
	if (format == "CSV")
	{
		QStringList header;
		if (singleKeyPointRows)
		{
			header << "x"
			       << "y"
			       << "size"
			       << "angle"
			       << "response"
			       << "octave"
			       << "class id";
		}
		else
		{
			header << "match distance"
			       << "img idx"
			       << "query idx"
			       << "train idx"
			       << "key point 1 x"
			       << "y 1"
			       << "size 1"
			       << "angle 1"
			       << "response 1"
			       << "octave 1"
			       << "class id 1"
			       << "key point 2 x"
			       << "y 2"
			       << "size 2"
			       << "angle 2"
			       << "response 2"
			       << "octave 2"
			       << "class id 2";
		}
		lines << header.join(",");
	}
	// These functions are simple macros
	auto intstr = [](int number)
	{ return QString::number(number); };
	auto floatstr = [](float number)
	{ return QString::number(number); };
	// combine a name and a value string
	auto combinestr = [format](QString name, QString value)
	{
		QString fmtStr = "";
		if (format == "JSON")
		{
			fmtStr = "\"%1\": %2";
		}
		else if (format == "PYTHON")
		{
			fmtStr = "'%1': %2";
		}
		else if (format == "RUBY")
		{
			fmtStr = "\"%1\" => %2";
		}
		return fmtStr.arg(name, value);
	};
	// data sets
	for (auto &row : rows)
	{
		if (format == "CSV" && !singleKeyPointRows)
		{
			QStringList line;
			line << floatstr(row.matchDistance())
			     << intstr(row.matchImgIdx())
			     << intstr(row.matchQueryIdx())
			     << intstr(row.matchTrainIdx())
			     << floatstr(row.keyPoint1XCoord())
			     << floatstr(row.keyPoint1YCoord())
			     << floatstr(row.keyPoint1Size())
			     << floatstr(row.keyPoint1Angle())
			     << floatstr(row.keyPoint1Response())
			     << intstr(row.keyPoint1Octave())
			     << floatstr(row.keyPoint1ClassId())
			     << floatstr(row.keyPoint2XCoord())
			     << floatstr(row.keyPoint2YCoord())
			     << floatstr(row.keyPoint2Size())
			     << floatstr(row.keyPoint2Angle())
			     << floatstr(row.keyPoint2Response())
			     << intstr(row.keyPoint2Octave())
			     << floatstr(row.keyPoint2ClassId());
			lines << line.join(",");
		}
		else if ((format == "JSON" || format == "PYTHON" ||
		          format == "RUBY") &&
		         !singleKeyPointRows)
		{
			QStringList match;
			match << combinestr("match distance",
			                    floatstr(row.matchDistance()))
			      << combinestr("img idx",
			                    intstr(row.matchImgIdx()))
			      << combinestr("query idx",
			                    intstr(row.matchQueryIdx()))
			      << combinestr("train idx",
			                    intstr(row.matchTrainIdx()));
			QStringList keyPoint1;
			keyPoint1
			    << combinestr("x", floatstr(row.keyPoint1XCoord()))
			    << combinestr("y", floatstr(row.keyPoint1YCoord()))
			    << combinestr("size", floatstr(row.keyPoint1Size()))
			    << combinestr("angle",
			                  floatstr(row.keyPoint1Angle()))
			    << combinestr("response",
			                  floatstr(row.keyPoint1Response()))
			    << combinestr("octave",
			                  intstr(row.keyPoint1Octave()))
			    << combinestr("class id",
			                  floatstr(row.keyPoint1ClassId()));
			QStringList keyPoint2;
			keyPoint2
			    << combinestr("x", floatstr(row.keyPoint2XCoord()))
			    << combinestr("y", floatstr(row.keyPoint2YCoord()))
			    << combinestr("size", floatstr(row.keyPoint2Size()))
			    << combinestr("angle",
			                  floatstr(row.keyPoint2Angle()))
			    << combinestr("response",
			                  floatstr(row.keyPoint2Response()))
			    << combinestr("octave",
			                  intstr(row.keyPoint2Octave()))
			    << combinestr("class id",
			                  floatstr(row.keyPoint2ClassId()));
			lines << QString("{%1, %2, %3}").arg(
			             combinestr("match", QString("{%1}").arg(
			                                     match.join(", "))),
			             combinestr("key point 1",
			                        QString("{%1}")
			                            .arg(keyPoint1.join(", "))),
			             combinestr("key point 2",
			                        QString("{%1}").arg(
			                            keyPoint2.join(", "))));
		}
		if (format == "CSV" && singleKeyPointRows)
		{
			QStringList line;
			line << floatstr(row.keyPoint1XCoord())
			     << floatstr(row.keyPoint1YCoord())
			     << floatstr(row.keyPoint1Size())
			     << floatstr(row.keyPoint1Angle())
			     << floatstr(row.keyPoint1Response())
			     << intstr(row.keyPoint1Octave())
			     << floatstr(row.keyPoint1ClassId())
			     << intstr(row.isLeftSingleKeyPoint() ? 1 : 2);
			lines << line.join(",");
		}
		else if ((format == "JSON" || format == "PYTHON" ||
		          format == "RUBY") &&
		         singleKeyPointRows)
		{
			QStringList keyPoint;
			keyPoint
			    << combinestr("x", floatstr(row.keyPoint1XCoord()))
			    << combinestr("y", floatstr(row.keyPoint1YCoord()))
			    << combinestr("size", floatstr(row.keyPoint1Size()))
			    << combinestr("angle",
			                  floatstr(row.keyPoint1Angle()))
			    << combinestr("response",
			                  floatstr(row.keyPoint1Response()))
			    << combinestr("octave",
			                  intstr(row.keyPoint1Octave()))
			    << combinestr("class id",
			                  floatstr(row.keyPoint1ClassId()))
			    << combinestr(
			           "img number",
			           intstr(row.isLeftSingleKeyPoint() ? 1 : 2));
			lines << QString("{%1}").arg(keyPoint.join(", "));
		}
	}
	QString text = "";
	// join the data sets with the header to a single block of text
	if (format == "CSV")
	{
		text = lines.join("\r\n"); // see RFC 4180
	}
	else if (format == "JSON" || format == "PYTHON" || format == "RUBY")
	{
		text = QString("[%1]").arg(lines.join(",\r\n"));
	}
	return text;
}

std::vector<QString> RawviewTableRow::getAvailableTextFormats()
{
	return { QString("CSV"),    QString("JSON"),
		 QString("PYTHON"), QString("RUBY") };
}

std::pair<QList<RawviewTableRow>, QList<RawviewTableRow>>
createRawviewTableRows(const std::vector<cv::KeyPoint> &keyPoints1,
                       const std::vector<cv::KeyPoint> &keyPoints2,
                       const std::vector<cv::DMatch> &matches,
					   bool usesTrainDescriptor)
{
	QList<RawviewTableRow> matchRowList;
	QList<RawviewTableRow> singleRowList;
	std::set<size_t> usedKeyPoints1;
	std::set<size_t> usedKeyPoints2;
	for (auto &match : matches)
	{
		int leftIndex = match.queryIdx;
		int rightIndex = usesTrainDescriptor ? match.trainIdx : match.imgIdx;
		if (leftIndex >= (int)keyPoints1.size() || rightIndex >= (int)keyPoints2.size())
		{
			continue;
		}
		usedKeyPoints1.insert(match.queryIdx);
		usedKeyPoints2.insert(match.trainIdx);
		matchRowList.append(
		    RawviewTableRow(match, keyPoints1.at(leftIndex),
		                    keyPoints2.at(rightIndex)));
	}
	for (size_t i = 0; i < usedKeyPoints1.size(); i++)
	{
		if (usedKeyPoints1.find(i) != usedKeyPoints1.end())
		{
			singleRowList.push_back(
			    RawviewTableRow(keyPoints1.at(i), true));
		}
	}
	for (size_t i = 0; i < usedKeyPoints2.size(); i++)
	{
		if (usedKeyPoints2.find(i) != usedKeyPoints2.end())
		{
			singleRowList.push_back(
			    RawviewTableRow(keyPoints2.at(i), false));
		}
	}
	return std::make_pair(matchRowList, singleRowList);
}

QList<RawviewTableRow>
createSingleKeyPointRawviewTableRows(const std::vector<cv::KeyPoint> &keyPoints,
                                     bool left)
{
	auto retList = QList<RawviewTableRow>();
	for (auto &keyPoint : keyPoints)
	{
		retList.append(RawviewTableRow(keyPoint, left));
	}
	return retList;
}
}
}

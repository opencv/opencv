#include "stringutils.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>


namespace cvv
{
namespace stfl
{

int stringEquality(const QString &str1, const QString &str2)
{
	if (isSingleWord(str1) && isSingleWord(str2))
	{
		return phoneticEquality(str1, str2);
	}
	return editDistance(str1, str2);
}

size_t editDistance(const QString &str1, const QString &str2)
{
	const unsigned len1 = str1.size();
	const unsigned len2 = str2.size();

	std::vector<size_t> col(len2 + 1);
	std::vector<size_t> prevCol(len2 + 1);

	// fills the vector with ascending numbers, starting by 0
	std::iota(prevCol.begin(), prevCol.end(), 0);

	for (unsigned i = 0; i < len1; i++)
	{
		col[0] = i + 1;
		for (unsigned j = 0; j < len2; j++)
		{
			if (str1[i] == str2[j])
				col[j + 1] =
				    std::min({ 1 + col[j], 1 + prevCol[1 + j],
					       prevCol[j] });
			else
				col[j + 1] =
				    std::min({ 1 + col[j], 1 + prevCol[1 + j],
					       prevCol[j] + 1 });
		}
		std::swap(col, prevCol);
	}
	return prevCol[len2];
}

int phoneticEquality(const QString &word1, const QString &word2)
{
	if (word1 == word2)
	{
		return 0;
	}
	return editDistance(nysiisForWord(word1), nysiisForWord(word2)) + 1;
}

QString nysiisForWord(QString word)
{
	static std::map<QString, QString> replacements = { { "MAC", "MCC" },
		                                           { "KN", "NN" },
		                                           { "K", "C" },
		                                           { "PH", "FF" },
		                                           { "PF", "FF" },
		                                           { "SCH", "SSS" } };
	static std::map<QString, QString> replacements2 = { { "EE", "Y" },
		                                            { "IE", "Y" },
		                                            { "DT", "D" },
		                                            { "RT", "D" },
		                                            { "NT", "D" },
		                                            { "ND", "D" } };
	static std::map<QString, QString> replacements3 = { { "EV", "AF" },
		                                            { "Ü", "A" },
		                                            { "Ö", "A" },
		                                            { "Ä", "A" },
		                                            { "O", "G" },
		                                            { "Z", "S" },
		                                            { "M", "N" },
		                                            { "KN", "N" },
		                                            { "K", "C" },
		                                            { "SCH", "SSS" },
		                                            { "PH", "FF" } };

	if (word.isEmpty())
	{
		return "";
	}

	QString code;
	word = word.toUpper();

	replaceIfStartsWith(word, replacements);

	replaceIfEndsWith(word, replacements2);
	code.append(word[0]);
	word = word.right(word.size() - 1);

	while (word.size() > 0)
	{
		if (isVowel(word[0]))
			word[0] = QChar('A');
		replaceIfStartsWith(word, replacements);
		if (!(word.startsWith("H") &&
		      (!isVowel(code[code.size() - 1]) ||
		       (word.size() >= 2 && !isVowel(word[1])))) &&
		    !(word.startsWith("W") && isVowel(code[code.size() - 1])))
		{
			if (word[0] != code[code.size() - 1])
			{
				code.append(word[0]);
			}
		}
		word = word.right(word.size() - 1);
	}
	if (code.endsWith("S"))
	{
		code = code.left(code.size() - 1);
	}
	if (code.endsWith("AY"))
	{
		code = code.right(code.size() - 1);
		code[code.size() - 1] = QChar('Y');
	}
	else if (code.endsWith("A"))
	{
		code = code.left(code.size() - 1);
	}
	code = removeRepeatedCharacters(code);
	return code;
}

QString nysiisForWordCached(const QString &word)
{
	static std::map<QString, QString> cache;
	if (word.isEmpty())
		return "";
	if (cache.count(word))
	{
		return cache[word];
	}
	else
	{
		QString code = nysiisForWord(word);
		cache[word] = code;
		return code;
	}
}

QString removeRepeatedCharacters(const QString &str)
{
	if (str.isEmpty())
	{
		return "";
	}
	QString res;
	res += str[0];
	auto iterator = str.begin();
	iterator++;
	std::copy_if(str.begin(), str.end(), std::back_inserter(res),
	             [res](QChar c)
	{ return c != res[res.size() - 1]; });
	return res;
}

void replaceIfStartsWith(QString &str, const QString &search,
                         const QString &replacement)
{
	if (str.startsWith(search))
	{
		if (search.size() == replacement.size())
		{
			for (int i = 0; i < replacement.size(); i++)
			{
				str[i] = replacement[i];
			}
		}
		else
		{
			str = str.right(str.size() - search.size())
			          .prepend(replacement);
		}
	}
}

void replaceIfStartsWith(QString &word,
                         const std::map<QString, QString> &replacements)
{
	for (auto iterator = replacements.begin();
	     iterator != replacements.end(); iterator++)
	{
		replaceIfStartsWith(word, iterator->first, iterator->second);
	}
}

void replaceIfEndsWith(QString &str, const QString &search,
                       const QString &replacement)
{
	if (str.endsWith(search))
	{
		if (search.length() == replacement.length())
		{
			for (int i = str.length() - replacement.length();
			     i < str.length(); i++)
			{
				str[i] = replacement[i];
			}
		}
		else
		{
			str = str.left(str.length() - search.length())
			          .append(replacement);
		}
	}
}

void replaceIfEndsWith(QString &word,
                       const std::map<QString, QString> &replacements)
{
	for (auto iterator = replacements.begin();
	     iterator != replacements.end(); iterator++)
	{
		replaceIfEndsWith(word, iterator->first, iterator->second);
	}
}

bool isVowel(const QChar &someChar)
{
	static std::vector<QChar> vowels = { 'a', 'e', 'i', 'o', 'u' };
	return std::find(vowels.begin(), vowels.end(), someChar) !=
	       vowels.end();
}

bool isSingleWord(const QString &str)
{
	const auto isLetter = [](QChar c)
	{ return c.isLetter(); };
	return std::find_if_not(str.begin(), str.end(), isLetter) != str.end();
}

void unescapeCommas(QString &str)
{
	str.replace("\\,", ",");
}

QString shortenString(QString &str, int maxLength, bool cutEnd, bool fill)
{
	if (str.size() > maxLength)
	{
		if (cutEnd)
		{
			str = str.mid(0, maxLength - 1) + u8"…";
		}
		else
		{
			str = u8"…" +
			      str.mid(str.size() + 1 - maxLength, str.size());
		}
	}
	else if (fill)
	{
		str = str + QString(maxLength - str.size(), ' ');
	}
	return str;
}

QString asciiCharVectorToQString(std::vector<char> chars)
{
	return QString::fromStdString(std::string(chars.begin(), chars.end()));
}
}
}

#ifndef CVVISUAL_STRINGUTILS_HPP
#define CVVISUAL_STRINGUTILS_HPP

#include <map>
#include <vector>

#include <QString>
#include <QHash>
#include <QRegExp>

namespace cvv
{
namespace stfl
{

/**
 * @brief Calculates the equality of two strings.
 * If both strings are only single words, a combination of the
 * levenshtein edit distance and a phonetic matching algorithm is used.
 * If not only the first is used.
 * Attention: using a phonetic algorithm is much slower, than the simple
 * levenshtein.
 * @param str1 first string
 * @param str2 second string
 * @return equality of both strings, 0 means both string are equal,
 *  the greater the number, the more unequal are both strings.
 */
int stringEquality(const QString &str1, const QString &str2);

/**
 * @brief Implementation of the levenshtein distance.
 * The levenshtein distance is a metric for the edit distance between to
 * strings.
 * Based on
 * http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#C.2B.2B
 * @param str1 first string
 * @param str2 second string
 * @return edit distance
 */
size_t editDistance(const QString &str1, const QString &str2);

/**
 * @brief Implementation of a phonetic algorithm to compare two words.
 * It generates the NYSIIS for both words and returns the levenshtein
 * edit distance between them.
 * @attention using a phonetic algorithm is much slower, than the simple
 * levenshtein,
 * and also consumes much more memory as it uses the cached version the NYSIIS
 * algorithm
 * @param word1 first word
 * @param word2 second word
 * @return equality of both words, 0 means both words are equal,
 * the greater the number, the more unequal are both words.
 */
int phoneticEquality(const QString &word1, const QString &word2);

/**
 * @brief Examines the NYSIIS of the given word.
 * The NYSIIS is the New York State Identification and Intelligence System
 * Phonetic Code,
 * http://en.wikipedia.org/wiki/NYSIIS.
 * @param word given word
 * @return NYSIIS of the given word
 */
QString nysiisForWord(QString word);

/**
 * @brief Examines the NYSIIS of the given word and caches it.
 * It's faster than the uncached method, at the cost of consuming more memory.
 * @param word given word
 * @return NYSIIS of the given word
 * @see nysiisForWord
 */
QString nysiisForWordCached(const QString &word);

/**
 * @brief Removes repeated chars in the given string.
 * E.g. "Hello World!!!" => "Helo World!"
 * @param str given string
 * @return resulting string
 */
QString removeRepeatedCharacters(const QString &str);

/**
 * @brief Replace the search string with its replacement at the very beginning
 * of the given string.
 * @param str given string
 */
void replaceIfStartsWith(QString &str, const QString &search,
                         const QString &replacement);

/**
 * @brief Replace the replacements at the very beginning of the given string.
 * Replace the key of the replacements map with its map value in the given
 * string
 * @param str given string
 * @param replacements replacements map
 */
void replaceIfStartsWith(QString &str,
                         const std::map<QString, QString> &replacements);

/**
 * @brief Replace the search string with its replacement at the end of the given
 * string.
 * @param str given string
 */
void replaceIfEndsWith(QString &str, const QString &search,
                       const QString &replacement);

/**
 * @brief Replace the replacements at the end of the given string.
 * Replace the key of the replacements map with its map value in the given
 * string
 * @param str given string
 * @param replacements replacements map
 */
void replaceIfEndsWith(QString &str,
                       const std::map<QString, QString> &replacements);

/**
 * Check whether or not the given char is a vowel.
 */
bool isVowel(const QChar &someChar);

/**
 * @brief Check wether the given string is a single word.
 * A word consists only of letters.
 *
 * @param str string to ckeck
 */
bool isSingleWord(const QString &str);

/**
 * @brief Unescapes escaped commas in the given string.
 *
 * @param str given string
 */
void unescapeCommas(QString &str);

/**
 * @brief Shortens the given string to the given length and append "..." if
 * needed.
 * @param str given string
 * @param maxLength maximum length of the returned string
 * @param cutEnd does this method shorten the given string at the end?
 * @param fill should the resulting string be filled up with whitespace to
 * ensure all strings have length maxLength?
 */
QString shortenString(QString &str, int maxLength, bool cutEnd = true,
                      bool fill = false);

/**
 * @brief Converts a given vector of chars into a valid QString.
 * @param chars given vector of chars
 * @return resulting QString
 */
QString asciiCharVectorToQString(std::vector<char> chars);
}
}

#endif

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include <sstream>
#include <iostream>
#include <regex>
#include "opencv2/core/check.hpp"

using std::cout;
using std::endl;
using std::for_each;
using std::string;
using std::vector;
using namespace std::placeholders;

namespace {

static const string magic_none_value = "<none>";

inline static string trim(const string& str)
{
    const string::size_type first_nonspace = str.find_first_not_of(" \t\n\r");
    const string::size_type last_nonspace = str.find_last_not_of(" \t\n\r");
    if (first_nonspace == string::npos || last_nonspace == string::npos)
        return string();
    return str.substr(first_nonspace, last_nonspace - first_nonspace + 1);
}

inline static string trim1(const string& str)
{
    return str.substr(1, str.size() - 1);
}

inline static std::vector<string> split_string(string str, char symbol)
{
    vector<string> vec;
    string::size_type pos = str.find_first_not_of(symbol);
    while (pos != string::npos)
    {
        const string::size_type end_pos = str.find_first_of(symbol, pos);
        if (end_pos != string::npos)
        {
            vec.push_back(str.substr(pos, end_pos - pos));
            pos = str.find_first_not_of(symbol, end_pos);
            continue;
        }
        vec.push_back(str.substr(pos, string::npos));
        break;
    }
    return vec;
}

} // namespace

struct CommandLineParserParams
{
public:
    vector<string> keys;
    string value;
    string help_message;
    int number;
    bool isPositional;
    bool isValid;
public:
    CommandLineParserParams() : number(-1), isPositional(false), isValid(false) {}
    CommandLineParserParams(const vector<string>& keys_, const string& def_value_, const string& help_message_)
        : keys(keys_), value(def_value_), help_message(trim(help_message_)),
          number(-1), isValid(true)
    {
        CV_Assert(!keys.empty());
        isPositional = keys[0].find('@') == 0;
    }
    inline string getPrintedKey() const
    {
        if (isPositional)
            return trim1(keys[0]);
        std::ostringstream buf;
        for (std::vector<string>::const_iterator key = keys.begin(); key != keys.end(); ++key)
        {
            buf << (key->length() > 1 ? "--" : "-") << *key;
            if (key + 1 != keys.end())
                buf << ", ";
        }
        return buf.str();
    }
    inline string getPrintedValue() const
    {
        const string def = trim(value);
        if (!def.empty())
            return " (value:" + def + ")";
        return string();
    }
    inline string getValue(bool trim_spaces = true) const
    {
        return trim_spaces ? trim(value) : value;
    }
    bool operator<(const CommandLineParserParams & other) const
    {
        if (isPositional == other.isPositional)
            if (number == other.number)
                return keys[0] < other.keys[0];
            else
                return number < other.number;
        else
            return isPositional < other.isPositional;
    }
};

//==================================================================================================

namespace cv
{

struct CommandLineParser::Impl
{
private:
    bool error;
    std::ostringstream error_message;
    string about_message;
    string path_to_app;
    string app_name;
    std::vector<CommandLineParserParams> data;
    CommandLineParserParams invalid;

    typedef vector<CommandLineParserParams>::iterator DataIterator;
    typedef vector<CommandLineParserParams>::const_iterator ConstDataIterator;
    typedef vector<string>::const_iterator KeysIterator;
public:
    Impl(const std::string & keys) : error(false)
    {
        // parse keys
        const std::regex rx("\\{"
                            "([^\\{\\}\\|]*)" "\\|"
                            "([^\\{\\}\\|]*)" "\\|"
                            "([^\\{\\}\\|]*)"
                            "\\}");
        auto match_begin = std::sregex_iterator(keys.begin(), keys.end(), rx);
        auto match_end = std::sregex_iterator();
        int positional_counter = 0;
        for (std::sregex_iterator i = match_begin; i != match_end; ++i)
        {
            const std::smatch & pieces = *i;
            const vector<string> parsed_keys = split_string(pieces[1], ' ');
            CV_CheckGT(parsed_keys.size(), (size_t)0, "Field KEYS can't be empty!");
            CommandLineParserParams p(parsed_keys, pieces[2], pieces[3]);
            if (p.isPositional)
                p.number = positional_counter++;
            data.push_back(p);
        }
    }
    void parse(int argc, const char * const argv[])
    {
        if (argc <= 0)
            return;
        // path to application
        const string appName(argv[0]);
        const string::size_type pos_s = appName.find_last_of("/\\");
        if (pos_s == string::npos)
        {
            path_to_app = "";
            app_name = appName;
        }
        else
        {
            path_to_app = appName.substr(0, pos_s);
            app_name = appName.substr(pos_s + 1, appName.size() - pos_s);
        }

        // parse argv
        const std::regex bool_rx("--?([^=]+)=?");
        const std::regex val_rx("--?([^=]+)=(.+)");
        int positional_counter = 0;
        for (int i = 1; i < argc; i++)
        {
            const string one_arg(argv[i]);
            std::smatch match;
            if (std::regex_match(one_arg, match, bool_rx))
                apply_params(match[1], "true");
            else if (std::regex_match(one_arg, match, val_rx))
                apply_params(match[1], match[2]);
            else
                apply_params(positional_counter++, one_arg);
        }
        sort_params();
    }
    CommandLineParserParams& findByName(const std::string& name)
    {
        for (DataIterator param = data.begin(); param != data.end(); ++param)
            for (KeysIterator key = param->keys.begin(); key != param->keys.end(); ++key)
                if (name == *key)
                    return *param;
        return invalid;
    }
    CommandLineParserParams& findByIndex(int index)
    {
        for (DataIterator param = data.begin(); param != data.end(); ++param)
            if (param->number == index)
                return *param;
        return invalid;
    }
    void apply_params(const String& key, const String& value)
    {
        CommandLineParserParams& param = findByName(key);
        if (!param.isValid)
            return;
        param.value = value;
    }
    void apply_params(int i, String value)
    {
        CommandLineParserParams& param = findByIndex(i);
        if (!param.isValid)
            return;
        param.value = value;
    }
    void sort_params()
    {
        for (size_t i = 0; i < data.size(); i++)
            std::sort(data[i].keys.begin(), data[i].keys.end());
        std::sort(data.begin(), data.end());
    }
    std::ostream& addError()
    {
        error = true;
        return error_message;
    }
    bool isGood() const
    {
        return !error;
    }
    string getErrorMessage() const
    {
        return error_message.str();
    }

    void setAboutMessage(const std::string& msg)
    {
        about_message = msg;
    }
    string getApplicationPath() const
    {
        return path_to_app;
    }
    void printMessage(std::ostream& out) const
    {
        if (!about_message.empty())
            out << about_message << endl;
        out << "Usage: " << app_name << " [params] ";
        for (ConstDataIterator i = data.begin(); i != data.end(); ++i)
        {
            if (i->isPositional)
                out << i->getPrintedKey() << " ";
        }
        out << endl << endl;
        for (ConstDataIterator i = data.begin(); i != data.end(); ++i)
        {
            out << '\t' << i->getPrintedKey() << i->getPrintedValue() << endl;
            out << '\t' << '\t' << i->help_message << endl;
        }
    }
};

} // namespace cv

//==================================================================================================

namespace {

template<typename T>
inline bool from_str(const string& line, void* dst)
{
    std::istringstream input(line);
    input.imbue(std::locale::classic());
    T val;
    input >> val;
    if (input.fail())
        return false;
    memcpy(dst, &val, sizeof(T));
    return true;
}

template<>
inline bool from_str<bool>(const string& line, void* dst)
{
    std::istringstream input(line);
    input.imbue(std::locale::classic());
    string str;
    input >> str;
    str = cv::toLowerCase(str);
    bool val = false;
    if (str == "false" || str == "0")
        val = false;
    else if (str == "true" || str == "1")
        val = true;
    else
        input.setstate(std::istream::failbit);
    if (input.fail())
        return false;
    memcpy(dst, &val, sizeof(bool));
    return true;
}

template<>
inline bool from_str<cv::Scalar>(const string& line, void* dst)
{
    std::istringstream input(line);
    input.imbue(std::locale::classic());
    cv::Scalar res;
    int i = 0;
    while (!input.eof() && i < 4)
        input >> res.val[i++];
    if (input.fail())
        return false;
    memcpy(dst, &res, sizeof(cv::Scalar));
    return true;
}

inline static bool from_str(const string& str, cv::Param type, void* dst)
{
    switch(type)
    {
    case cv::Param::INT: return from_str<int>(str, dst);
    case cv::Param::BOOLEAN: return from_str<bool>(str, dst);
    case cv::Param::UNSIGNED_INT: return from_str<unsigned>(str, dst);
    case cv::Param::UINT64: return from_str<uint64>(str, dst);
    case cv::Param::FLOAT: return from_str<float>(str, dst);
    case cv::Param::REAL: return from_str<double>(str, dst);
    case cv::Param::SCALAR: return from_str<cv::Scalar>(str, dst);
    case cv::Param::STRING:
        static_cast<string*>(dst)->assign(str);
        return true;
    default: return false;
    }
}

} // namespace

//==================================================================================================

namespace cv
{

inline static std::ostream & operator<<(std::ostream& out, cv::Param type)
{
    switch(type)
    {
    case cv::Param::INT: out << "int"; break;
    case cv::Param::BOOLEAN: out << "bool"; break;
    case cv::Param::UNSIGNED_INT: out << "unsigned"; break;
    case cv::Param::UINT64: out << "unsigned long long"; break;
    case cv::Param::FLOAT: out << "float"; break;
    case cv::Param::REAL: out << "double"; break;
    case cv::Param::SCALAR: out << "scalar"; break;
    case cv::Param::STRING: out << "string"; break;
    default: out << "unknown"; break;
    }
    return out;
}

void CommandLineParser::getByName(const String& name, bool space_delete, Param type, void* dst) const
{
    const CommandLineParserParams& param = impl->findByName(name);
    if (!param.isValid)
        CV_Error(Error::StsBadArg, "Requested invalid parameter: '" + name + "'");
    const string value = param.getValue(space_delete);
    if (value == magic_none_value || (value.empty() && type != Param::STRING))
        impl->addError() << "Missing parameter: '" << name << "'" << endl;
    else if (!from_str(value, type, dst))
        impl->addError() << "Can not convert parameter '" << name << "': [" << value << "] to [" << type << "]" << endl;
}

void CommandLineParser::getByIndex(int index, bool space_delete, Param type, void* dst) const
{
    const CommandLineParserParams& param = impl->findByIndex(index);
    if (!param.isValid)
        CV_Error(Error::StsBadArg, "Requested invalid parameter: '" + format("%d", index) + "'");
    const string value = param.getValue(space_delete);
    if (value == magic_none_value || (value.empty() && type != Param::STRING))
        impl->addError() << "Missing parameter #" << index << endl;
    else if (!from_str(value, type, dst))
        impl->addError() << "Can not convert parameter #" << index << " [" << value << "] to [" << type << "]" << endl;
}

CommandLineParser::CommandLineParser(int argc, const char* const argv[], const String& keys)
{
    impl = new Impl(keys);
    impl->parse(argc, argv);
}

CommandLineParser::~CommandLineParser()
{
    delete impl;
}

void CommandLineParser::about(const String& message)
{
    impl->setAboutMessage(message);
}

String CommandLineParser::getPathToApplication() const
{
    return impl->getApplicationPath();
}

bool CommandLineParser::has(const String& name) const
{
    const CommandLineParserParams& param = impl->findByName(name);
    if (!param.isValid)
        CV_Error(Error::StsBadArg, "Requested invalid parameter: '" + name + "'");
    const string value = param.getValue();
    return value != magic_none_value && !value.empty();
}

bool CommandLineParser::check() const
{
    return impl->isGood();
}

void CommandLineParser::printErrors() const
{
    if (!impl->isGood())
        cout << endl << "ERRORS:" << endl << impl->getErrorMessage() << endl;
}

void CommandLineParser::printMessage() const
{
    impl->printMessage(cout);
}

} // namespace cv

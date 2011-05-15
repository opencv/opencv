#include "precomp.hpp"

using namespace std;
using namespace cv;


vector<string> split_string(const string& str, const string& delimiters)
{
    vector<string> res;
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos)
    {
        res.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }

    return res;
}

void PreprocessArgs(int _argc, const char* _argv[], int& argc, char**& argv)
{
    std::vector<std::string> buffer_vector;
    std::string buffer_string;
    std::string buffer2_string;
    int find_symbol;

    for (int i = 0; i < _argc; i++)
    {
        buffer_string = _argv[i];
        find_symbol = buffer_string.find('=');
        if (find_symbol == -1)
            buffer_vector.push_back(buffer_string);
        else if (find_symbol == 0 || find_symbol == (buffer_string.length() - 1))
        {
            buffer_string.erase(find_symbol, (find_symbol + 1));
            buffer_vector.push_back(buffer_string);
        }
        else
        {
            buffer2_string = buffer_string;
            buffer_string.erase(find_symbol);
            buffer_vector.push_back(buffer_string);
            buffer2_string.erase(0, find_symbol + 1);
            buffer_vector.push_back(buffer2_string);
        }
    }

    argc = buffer_vector.size();
    argv = new char* [argc];
    for (int i=0; i < argc; i++)
    {
        argv[i] = new char[buffer_vector[i].length() + 1];
        memcpy(argv[i], buffer_vector[i].c_str(), buffer_vector[i].length() + 1);
    }
}

CommandLineParser::CommandLineParser(int _argc, const char* _argv[])
{
    std::string cur_name;
    bool was_pushed=false;
    int argc;
    char** argv;

    PreprocessArgs(_argc, _argv, argc, argv);

    for(int i=1; i < argc; i++)
    {
        if(!argv[i])
            break;

        if( (argv[i][0]== '-') && (strlen(argv[i]) > 1) &&
            ((argv[i][1] < '0') || (argv[i][1] > '9'))   )
        {
            if (!cur_name.empty() && !was_pushed)
            {
                data[cur_name].push_back("");
            }
            cur_name=argv[i];
            was_pushed=false;

            if (data.find(cur_name) != data.end())
            {
                string str_exception = "dublicating parameters for name='" + cur_name + "'";
                CV_Error(CV_StsParseError, str_exception);
            }
            continue;
        }

        data[cur_name].push_back(argv[i]);
        was_pushed=true;
    }
    if (!cur_name.empty() && !was_pushed)
        data[cur_name].push_back("");
}

bool CommandLineParser::has(const std::string& keys) const
{
    vector<string> names=split_string(keys, " |");
    for(size_t j=0; j < names.size(); j++)
    {
        if (data.find(names[j])!=data.end())
            return true;
    }
    return false;
}

template<>
std::vector<std::string> CommandLineParser::getVec<std::string>(const std::string& keys)
{
    vector<string> names=split_string(keys, " |");

    int found_index=-1;
    for(size_t j=0; j < names.size(); j++)
    {
        const string& cur_name=names[j];
        bool is_cur_found=has(cur_name);

        if (is_cur_found && (found_index >= 0))
        {
            string str_exception = "dublicating parameters for "
                                   "name='" + names[found_index] + "' and name='"+cur_name+"'";
            CV_Error(CV_StsParseError, str_exception);
        }

        if (is_cur_found)
            found_index=j;
    }

    if (found_index<0)
        return vector<string>();

    return data.find(names[found_index])->second;
}

template<>
std::string CommandLineParser::fromString<std::string>(const std::string& str)
{
    return str;
}

template<>
int CommandLineParser::fromString<int>(const std::string& str)
{
    return fromStringNumber<int>(str);
}

template<>
unsigned int CommandLineParser::fromString<unsigned int>(const std::string& str)
{
    return fromStringNumber<unsigned int>(str);
}

template<>
double CommandLineParser::fromString<double>(const std::string& str)
{
    return fromStringNumber<double>(str);
}

template<>
cv::Size CommandLineParser::fromStringsVec<cv::Size>(const std::vector<std::string>& vec_str)
{
    if (vec_str.size() < 2)
        CV_Error(CV_StsParseError, "Cannot convert vector of string to cv::Size : less than two strings");

    cv::Size res;
    res.width=fromString<int>(vec_str[0]);
    res.height=fromString<int>(vec_str[1]);

    return res;
}

#include "precomp.hpp"

using namespace std;
using namespace cv;

void helpParser()
{
    printf("\nThe CommandLineParser class is designed for command line arguments parsing\n"
           "Supported syntax: \n"
           "    --key1=arg1 or --key3 <The keys with '--' can have argument.\n"
                                     "If it has argument, you should assign it through '=' sign> \n"
           "    -key2=arg2 or -key2  <The keys witn '-' can have argument \n"
                                    "If it has argument, you should assign it through '=' sign> \n"
           "    key3                 <This key can't has any parameter> \n"
           "Usage: \n"
           "      Imagine that the input parameters are next:\n"
           "                -k=10 --key --db=-10.11 -key1 argument --inputFile=lena.jpg\n"
           "parser.get<int>(\"k\")<If you need to take k value.\n"
           "                    It also works with 'unsigned int', 'double', 'float' and 'string' types>\n"
           "parser.get<double>(\"db\", 99.99)<If you need to take db value.\n"
           "                                If its value is empty, you will get default value 99.99>\n"
           "                                It also works with 'int', 'unsigned int', 'float' and 'string' types\n"
           "parser.get<bool>(\"key\")<The method return 'true', if 'key' was defined in command line\n"
           "                          and it will return 'false' otherwise.>\n"
           "parser.get<bool>(\"key1\")<The method return 'true', if 'key' was defined in command line\n"
           "                          and it will return 'false' otherwise.>\n"
           "parser.get<stirng>(\"0\")<If you need to take argument. It's the first parameter without '-' or '--' increment \n"
           "                          and without value. It has index 0. The second parameter of this type will have index 1>\n"
           "                          It also works with 'int', 'unsigned int', 'double' and 'float' types\n\n"
           );
}

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

CommandLineParser::CommandLineParser(int argc, const char* argv[])
{
    std::string cur_name;
    std::string buffer;
    std::stringstream str_buff(std::stringstream::in | std::stringstream::out);
    std::map<std::string, std::string >::iterator it;
    int find_symbol;
    int index = 0;


    for(int i = 1; i < argc; i++)
    {

        if(!argv[i])
            break;
        cur_name = argv[i];
        if((cur_name.find('-') == 0) && ((int)cur_name.find('=') != -1) &&
           (cur_name.find('=') != (cur_name.length() - 1)))
        {
            while (cur_name.find('-') == 0)
                cur_name.erase(0,1);

            buffer = cur_name;
            find_symbol = (int)cur_name.find('=');
            cur_name.erase(find_symbol);
            buffer.erase(0, find_symbol + 1);
            if (data.find(cur_name) != data.end())
            {
                printf("CommandLineParser constructor found dublicating parameters for name=%s\n"
                       , cur_name.c_str());
                printf("Constructor will not continue its work since this moment.\n"
                       "Please enter parameters without dublicates\n");
                helpParser();
                data.clear();
                break;
            }
                else
                    data[cur_name] = buffer;
        }
            else if (cur_name.find('=') == 0)
                {
                    printf("The next key is wrong: key= %s\n", cur_name.c_str());
                    printf("Constructor will not continue its work since this moment.\n"
                           "Please enter parameters without any mistakes\n");
                    helpParser();
                    data.clear();
                    break;
                }
            else if(((int)cur_name.find('-') == -1) && ((int)cur_name.find('=') != -1))
                {
                    printf("The next key must be defined with '--' or '-' increment: key= %s\n", cur_name.c_str());
                    printf("Constructor will not continue its work since this moment.\n"
                           "Please enter parameters without any mistakes\n");
                    helpParser();
                    data.clear();
                    break;
                }
            else if (cur_name.find('=') == (cur_name.length() - 1))
                {
                    printf("The next key must have argument after '=': key= %s\n", cur_name.c_str());
                    printf("Constructor will not continue its work since this moment.\n"
                           "Please enter parameters without any mistakes\n");
                    helpParser();
                    data.clear();
                    break;
                }
            else if ((cur_name.find('-') == 0) && ((cur_name[1] < '0') || (cur_name[1] > '9')) )
                {
                    while (cur_name.find('-') == 0)
                    cur_name.erase(0,1);

                    for(it = data.begin(); it != data.end(); it++)
                    {
                        if (it->first == cur_name)
                        {
                            printf("CommandLineParser constructor found dublicating parameters for name=%s\n"
                                   , cur_name.c_str());
                            printf("Constructor will not continue its work since this moment.\n"
                                   "Please enter parameters without dublicates\n");
                            helpParser();
                            data.clear();
                            break;
                        }
                    }
                    data[cur_name] = "true";
                }
            else
                {
                    str_buff << index;
                    for(it = data.begin(); it != data.end(); it++)
                    {
                        if (it->second == cur_name)
                        {
                            printf("CommandLineParser constructor found dublicating parameters for name=%s\n"
                                   , cur_name.c_str());
                            printf("Constructor will not continue its work since this moment.\n"
                                   "Please enter parameters without dublicates\n");
                            helpParser();
                            data.clear();
                            break;
                        }
                    }
                    data[str_buff.str()] = cur_name;
                    str_buff.seekp(0);
                    index++;
                }

    }
}

bool CommandLineParser::has(const std::string& keys) const
{
    vector<string> names=split_string(keys, " |");
    for(size_t j=0; j < names.size(); j++) {
        if (data.find(names[j])!=data.end())
            return true;
    }
    return false;
}

std::string CommandLineParser::getString(const std::string& keys) const
{
    vector<string> names=split_string(keys, " |");

    int found_index=-1;
    for(size_t j=0; j < names.size(); j++) {
        const string& cur_name=names[j];
        bool is_cur_found=has(cur_name);

        if (is_cur_found && (found_index >= 0)) {
            string str_exception="dublicating parameters for "
                "name='" + names[found_index] + "' and name='"+cur_name+"'";
            CV_Error(CV_StsParseError, str_exception);
        }

        if (is_cur_found)
            found_index=(int)j;
    }

    if (found_index<0)
        return string();
    return data.find(names[found_index])->second;
}

template<typename _Tp>
 _Tp CommandLineParser::fromStringNumber(const std::string& str) //the default conversion function for numbers
{
    if (str.empty())
        CV_Error(CV_StsParseError, "Empty string cannot be converted to a number");

    const char* c_str=str.c_str();
    if((!isdigit(c_str[0]))
        &&
        (
            (c_str[0]!='-') || (strlen(c_str) <= 1) || ( !isdigit(c_str[1]) )
        )
    )

    {
        CV_Error(CV_StsParseError, "The string '"+ str +"' cannot be converted to a number");
    }

    return  getData<_Tp>(str);
}

template<typename _Tp>
static _Tp getData(const std::string& str)
{
    _Tp res;
    std::stringstream s1(str);
    s1 >> res;
    return res;
}

template<typename _Tp>
static _Tp fromStringNumber(const std::string& str)//the default conversion function for numbers
{
    
    if (str.empty())
        CV_Error(CV_StsParseError, "Empty string cannot be converted to a number");
    
    const char* c_str=str.c_str();
    if( !isdigit(c_str[0]) &&
       (c_str[0] != '-' || strlen(c_str) <= 1 || !isdigit(c_str[1]) ))
        CV_Error(CV_StsParseError, "The string '"+ str +"' cannot be converted to a number");
    
    return  getData<_Tp>(str);
}

template<>
bool CommandLineParser::get<bool>(const std::string& name, const bool& /*default_value*/)
{
    if (!has(name))
        return false;
    return true;
}

template<>
std::string CommandLineParser::analyzeValue<std::string>(const std::string& str)
{
    return str;
}

template<>
int CommandLineParser::analyzeValue<int>(const std::string& str)
{
    return fromStringNumber<int>(str);
}

template<>
unsigned int CommandLineParser::analyzeValue<unsigned int>(const std::string& str)
{
    return fromStringNumber<unsigned int>(str);
}

template<>
float CommandLineParser::analyzeValue<float>(const std::string& str)
{
    return fromStringNumber<float>(str);
}

template<>
double CommandLineParser::analyzeValue<double>(const std::string& str)
{
    return fromStringNumber<double>(str);
}


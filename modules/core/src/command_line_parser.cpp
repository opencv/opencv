
#include "precomp.hpp"

#include <iostream>

namespace cv
{

struct CommandLineParserParams
{
public:
    string help_message;
    string def_value;
    vector<string> keys;
    int number;
};


struct CommandLineParser::Impl
{
    bool error;
    string error_message;
    string about_message;

    string path_to_app;
    string app_name;

    vector<CommandLineParserParams> data;

    vector<string> split_range_string(const string& str, char fs, char ss) const;
    vector<string> split_string(const string& str, char symbol = ' ', bool create_empty_item = false) const;
    string cat_string(const string& str) const;

    void apply_params(const string& key, const string& value);
    void apply_params(int i, string value);

    void sort_params();
    int refcount;
};


static string get_type_name(int type)
{
    if( type == Param::INT )
        return "int";
    if( type == Param::UNSIGNED_INT )
        return "unsigned";
    if( type == Param::UINT64 )
        return "unsigned long long";
    if( type == Param::FLOAT )
        return "float";
    if( type == Param::REAL )
        return "double";
    if( type == Param::STRING )
        return "string";
    return "unknown";
}

static void from_str(const string& str, int type, void* dst)
{
    std::stringstream ss(str);
    if( type == Param::INT )
        ss >> *(int*)dst;
    else if( type == Param::UNSIGNED_INT )
        ss >> *(unsigned*)dst;
    else if( type == Param::UINT64 )
        ss >> *(uint64*)dst;
    else if( type == Param::FLOAT )
        ss >> *(float*)dst;
    else if( type == Param::REAL )
        ss >> *(double*)dst;
    else if( type == Param::STRING )
        ss >> *(string*)dst;
    else
        throw cv::Exception(CV_StsBadArg, "unknown/unsupported parameter type", "", __FILE__, __LINE__);

    if (ss.fail())
    {
        string err_msg = "can not convert: [" + str +
        + "] to [" + get_type_name(type) + "]";

        throw cv::Exception(CV_StsBadArg, err_msg, "", __FILE__, __LINE__);
    }
}

void CommandLineParser::getByName(const string& name, bool space_delete, int type, void* dst) const
{
    try
    {
        for (size_t i = 0; i < impl->data.size(); i++)
        {
            for (size_t j = 0; j < impl->data[i].keys.size(); j++)
            {
                if (name.compare(impl->data[i].keys[j]) == 0)
                {
                    string v = impl->data[i].def_value;
                    if (space_delete)
                        v = impl->cat_string(v);
                    from_str(v, type, dst);
                    return;
                }
            }
        }
        impl->error = true;
        impl->error_message += "Unknown parametes " + name + "\n";
    }
    catch (std::exception& e)
    {
        impl->error = true;
        impl->error_message += "Exception: " + string(e.what()) + "\n";
    }
}


void CommandLineParser::getByIndex(int index, bool space_delete, int type, void* dst) const
{
    try
    {
        for (size_t i = 0; i < impl->data.size(); i++)
        {
            if (impl->data[i].number == index)
            {
                string v = impl->data[i].def_value;
                if (space_delete == true) v = impl->cat_string(v);
                from_str(v, type, dst);
                return;
            }
        }
        impl->error = true;
        impl->error_message += "Unknown parametes #" + format("%d", index) + "\n";
    }
    catch(std::exception & e)
    {
        impl->error = true;
        impl->error_message += "Exception: " + string(e.what()) + "\n";
    }
}

static bool cmp_params(const CommandLineParserParams & p1, const CommandLineParserParams & p2)
{
    if (p1.number > p2.number)
        return false;

    if (p1.number == -1 && p2.number == -1)
    {
        if (p1.keys[0].compare(p2.keys[0]) > 0)
        {
            return false;
        }
    }

    return true;
}

CommandLineParser::CommandLineParser(int argc, const char* const argv[], const string& keys)
{
    impl = new Impl;
    impl->refcount = 1;

    // path to application
    size_t pos_s = string(argv[0]).find_last_of("/\\");
    if (pos_s == string::npos)
    {
        impl->path_to_app = "";
        impl->app_name = string(argv[0]);
    }
    else
    {
        impl->path_to_app = string(argv[0]).substr(0, pos_s);
        impl->app_name = string(argv[0]).substr(pos_s + 1, string(argv[0]).length() - pos_s);
    }

    impl->error = false;
    impl->error_message = "";

    // parse keys
    vector<string> k = impl->split_range_string(keys, '{', '}');

    int jj = 0;
    for (size_t i = 0; i < k.size(); i++)
    {
        vector<string> l = impl->split_string(k[i], '|', true);
        CommandLineParserParams p;
        p.keys = impl->split_string(l[0]);
        p.def_value = l[1];
        p.help_message = impl->cat_string(l[2]);
        p.number = -1;
        if (p.keys[0][0] == '@')
        {
            p.number = jj;
            jj++;
        }

        impl->data.push_back(p);
    }

    // parse argv
    jj = 0;
    for (int i = 1; i < argc; i++)
    {
        string s = string(argv[i]);

        if (s.find('=') != string::npos && s.find('=') < s.length())
        {
            vector<string> k_v = impl->split_string(s, '=', true);
            for (int h = 0; h < 2; h++)
            {
                if (k_v[0][0] == '-')
                    k_v[0] = k_v[0].substr(1, k_v[0].length() -1);
            }
            impl->apply_params(k_v[0], k_v[1]);
        }
        else if (s.length() > 1 && s[0] == '-')
        {
            for (int h = 0; h < 2; h++)
            {
                if (s[0] == '-')
                    s = s.substr(1, s.length() - 1);
            }
            impl->apply_params(s, "true");
        }
        else if (s[0] != '-')
        {
            impl->apply_params(jj, s);
            jj++;
        }
    }

    impl->sort_params();
}


CommandLineParser::CommandLineParser(const CommandLineParser& parser)
{
    impl = parser.impl;
    CV_XADD(&impl->refcount, 1);
}

CommandLineParser& CommandLineParser::operator = (const CommandLineParser& parser)
{
    if( this != &parser )
    {
        if(CV_XADD(&impl->refcount, -1) == 1)
            delete impl;
        impl = parser.impl;
        CV_XADD(&impl->refcount, 1);
    }
    return *this;
}

void CommandLineParser::about(const string& message)
{
    impl->about_message = message;
}

void CommandLineParser::Impl::apply_params(const string& key, const string& value)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        for (size_t k = 0; k < data[i].keys.size(); k++)
        {
            if (key.compare(data[i].keys[k]) == 0)
            {
                data[i].def_value = value;
                break;
            }
        }
    }
}

void CommandLineParser::Impl::apply_params(int i, string value)
{
    for (size_t j = 0; j < data.size(); j++)
    {
        if (data[j].number == i)
        {
            data[j].def_value = value;
            break;
        }
    }
}

void CommandLineParser::Impl::sort_params()
{
    for (size_t i = 0; i < data.size(); i++)
    {
        sort(data[i].keys.begin(), data[i].keys.end());
    }

    sort (data.begin(), data.end(), cmp_params);
}

string CommandLineParser::Impl::cat_string(const string& str) const
{
    int left = 0, right = (int)str.length();
    while( left <= right && str[left] == ' ' )
        left++;
    while( right > left && str[right-1] == ' ' )
        right--;
    return left >= right ? string("") : str.substr(left, right-left);
}

string CommandLineParser::getPathToApplication() const
{
    return impl->path_to_app;
}

bool CommandLineParser::has(const string& name) const
{
    for (size_t i = 0; i < impl->data.size(); i++)
    {
        for (size_t j = 0; j < impl->data[i].keys.size(); j++)
        {
            if (name.compare(impl->data[i].keys[j]) == 0 && string("true").compare(impl->data[i].def_value) == 0)
            {
                return true;
            }
        }
    }
    return false;
}

bool CommandLineParser::check() const
{
    return impl->error == false;
}

void CommandLineParser::printErrors() const
{
    if (impl->error)
    {
        std::cout << std::endl << "ERRORS:" << std::endl << impl->error_message << std::endl;
    }
}

void CommandLineParser::printMessage() const
{
    if (impl->about_message != "")
        std::cout << impl->about_message << std::endl;

    std::cout << "Usage: " << impl->app_name << " [params] ";

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number > -1)
        {
            string name = impl->data[i].keys[0].substr(1, impl->data[i].keys[0].length() - 1);
            std::cout << name << " ";
        }
    }

    std::cout << std::endl << std::endl;

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number == -1)
        {
            std::cout << "\t";
            for (size_t j = 0; j < impl->data[i].keys.size(); j++)
            {
                string k = impl->data[i].keys[j];
                if (k.length() > 1)
                {
                    std::cout << "--";
                }
                else
                {
                    std::cout << "-";
                }
                std::cout << k;

                if (j != impl->data[i].keys.size() - 1)
                {
                    std::cout << ", ";
                }
            }
            string dv = impl->cat_string(impl->data[i].def_value);
            if (dv.compare("") != 0)
            {
                std::cout << " (value:" << dv << ")";
            }
            std::cout << std::endl << "\t\t" << impl->data[i].help_message << std::endl;
        }
    }
    std::cout << std::endl;

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number != -1)
        {
            std::cout << "\t";
            string k = impl->data[i].keys[0];
            k = k.substr(1, k.length() - 1);

            std::cout << k;

            string dv = impl->cat_string(impl->data[i].def_value);
            if (dv.compare("") != 0)
            {
                std::cout << " (value:" << dv << ")";
            }
            std::cout << std::endl << "\t\t" << impl->data[i].help_message << std::endl;
        }
    }
}

vector<string> CommandLineParser::Impl::split_range_string(const string& _str, char fs, char ss) const
{
    string str = _str;
    vector<string> vec;
    string word = "";
    bool begin = false;

    while (!str.empty())
    {
        if (str[0] == fs)
        {
            if (begin == true)
            {
                throw cv::Exception(CV_StsParseError,
                         string("error in split_range_string(")
                         + str
                         + string(", ")
                         + string(1, fs)
                         + string(", ")
                         + string(1, ss)
                         + string(")"),
                         "", __FILE__, __LINE__
                         );
            }
            begin = true;
            word = "";
            str = str.substr(1, str.length() - 1);
        }

        if (str[0] == ss)
        {
            if (begin == false)
            {
                throw cv::Exception(CV_StsParseError,
                         string("error in split_range_string(")
                         + str
                         + string(", ")
                         + string(1, fs)
                         + string(", ")
                         + string(1, ss)
                         + string(")"),
                         "", __FILE__, __LINE__
                         );
            }
            begin = false;
            vec.push_back(word);
        }

        if (begin == true)
        {
            word += str[0];
        }
        str = str.substr(1, str.length() - 1);
    }

    if (begin == true)
    {
        throw cv::Exception(CV_StsParseError,
                 string("error in split_range_string(")
                 + str
                 + string(", ")
                 + string(1, fs)
                 + string(", ")
                 + string(1, ss)
                 + string(")"),
                 "", __FILE__, __LINE__
                );
    }

    return vec;
}

vector<string> CommandLineParser::Impl::split_string(const string& _str, char symbol, bool create_empty_item) const
{
    string str = _str;
    vector<string> vec;
    string word = "";

    while (!str.empty())
    {
        if (str[0] == symbol)
        {
            if (!word.empty() || create_empty_item)
            {
                vec.push_back(word);
                word = "";
            }
        }
        else
        {
            word += str[0];
        }
        str = str.substr(1, str.length() - 1);
    }

    if (word != "" || create_empty_item)
    {
        vec.push_back(word);
    }

    return vec;
}

}

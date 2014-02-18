#include "precomp.hpp"
#include <sstream>

namespace cv
{

struct CommandLineParserParams
{
public:
    String help_message;
    String def_value;
    std::vector<String> keys;
    int number;
};


struct CommandLineParser::Impl
{
    bool error;
    String error_message;
    String about_message;

    String path_to_app;
    String app_name;

    std::vector<CommandLineParserParams> data;

    std::vector<String> split_range_string(const String& str, char fs, char ss) const;
    std::vector<String> split_string(const String& str, char symbol = ' ', bool create_empty_item = false) const;
    String cat_string(const String& str) const;

    void apply_params(const String& key, const String& value);
    void apply_params(int i, String value);

    void sort_params();
    int refcount;
};


static String get_type_name(int type)
{
    if( type == Param::INT )
        return "int";
    if( type == Param::BOOLEAN )
        return "bool";
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

static void from_str(const String& str, int type, void* dst)
{
    std::stringstream ss(str.c_str());
    if( type == Param::INT )
        ss >> *(int*)dst;
    else if( type == Param::BOOLEAN )
    {
        std::string temp;
        ss >> temp;
        *(bool*) dst = temp == "true";
    }
    else if( type == Param::UNSIGNED_INT )
        ss >> *(unsigned*)dst;
    else if( type == Param::UINT64 )
        ss >> *(uint64*)dst;
    else if( type == Param::FLOAT )
        ss >> *(float*)dst;
    else if( type == Param::REAL )
        ss >> *(double*)dst;
    else if( type == Param::STRING )
        *(String*)dst = str;
    else
        throw cv::Exception(CV_StsBadArg, "unknown/unsupported parameter type", "", __FILE__, __LINE__);

    if (ss.fail())
    {
        String err_msg = "can not convert: [" + str +
        + "] to [" + get_type_name(type) + "]";

        throw cv::Exception(CV_StsBadArg, err_msg, "", __FILE__, __LINE__);
    }
}

void CommandLineParser::getByName(const String& name, bool space_delete, int type, void* dst) const
{
    try
    {
        for (size_t i = 0; i < impl->data.size(); i++)
        {
            for (size_t j = 0; j < impl->data[i].keys.size(); j++)
            {
                if (name.compare(impl->data[i].keys[j]) == 0)
                {
                    String v = impl->data[i].def_value;
                    if (space_delete)
                        v = impl->cat_string(v);
                    from_str(v, type, dst);
                    return;
                }
            }
        }
        impl->error = true;
        impl->error_message = impl->error_message + "Unknown parametes " + name + "\n";
    }
    catch (std::exception& e)
    {
        impl->error = true;
        impl->error_message = impl->error_message + "Exception: " + String(e.what()) + "\n";
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
                String v = impl->data[i].def_value;
                if (space_delete == true) v = impl->cat_string(v);
                from_str(v, type, dst);
                return;
            }
        }
        impl->error = true;
        impl->error_message = impl->error_message + "Unknown parametes #" + format("%d", index) + "\n";
    }
    catch(std::exception & e)
    {
        impl->error = true;
        impl->error_message = impl->error_message + "Exception: " + String(e.what()) + "\n";
    }
}

static bool cmp_params(const CommandLineParserParams & p1, const CommandLineParserParams & p2)
{
    if (p1.number < p2.number)
        return true;

    if (p1.number > p2.number)
        return false;

    return p1.keys[0].compare(p2.keys[0]) < 0;
}

CommandLineParser::CommandLineParser(int argc, const char* const argv[], const String& keys)
{
    impl = new Impl;
    impl->refcount = 1;

    // path to application
    size_t pos_s = String(argv[0]).find_last_of("/\\");
    if (pos_s == String::npos)
    {
        impl->path_to_app = "";
        impl->app_name = String(argv[0]);
    }
    else
    {
        impl->path_to_app = String(argv[0]).substr(0, pos_s);
        impl->app_name = String(argv[0]).substr(pos_s + 1, String(argv[0]).length() - pos_s);
    }

    impl->error = false;
    impl->error_message = "";

    // parse keys
    std::vector<String> k = impl->split_range_string(keys, '{', '}');

    int jj = 0;
    for (size_t i = 0; i < k.size(); i++)
    {
        std::vector<String> l = impl->split_string(k[i], '|', true);
        CommandLineParserParams p;
        p.keys = impl->split_string(l[0]);
        p.def_value = l[1];
        p.help_message = impl->cat_string(l[2]);
        p.number = -1;
        if (p.keys.size() <= 0)
        {
            impl->error = true;
            impl->error_message = "Field KEYS could not be empty\n";
        }
        else
        {
            if (p.keys[0][0] == '@')
            {
                p.number = jj;
                jj++;
            }

            impl->data.push_back(p);
        }
    }

    // parse argv
    jj = 0;
    for (int i = 1; i < argc; i++)
    {
        String s = String(argv[i]);

        if (s.find('=') != String::npos && s.find('=') < s.length())
        {
            std::vector<String> k_v = impl->split_string(s, '=', true);
            for (int h = 0; h < 2; h++)
            {
                if (k_v[0][0] == '-')
                    k_v[0] = k_v[0].substr(1, k_v[0].length() -1);
            }
            impl->apply_params(k_v[0], k_v[1]);
        }
        else if (s.length() > 2 && s[0] == '-' && s[1] == '-')
        {
            impl->apply_params(s.substr(2), "true");
        }
        else if (s.length() > 1 && s[0] == '-')
        {
            impl->apply_params(s.substr(1), "true");
        }
        else
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

void CommandLineParser::about(const String& message)
{
    impl->about_message = message;
}

void CommandLineParser::Impl::apply_params(const String& key, const String& value)
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

void CommandLineParser::Impl::apply_params(int i, String value)
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
        std::sort(data[i].keys.begin(), data[i].keys.end());
    }

    std::sort (data.begin(), data.end(), cmp_params);
}

String CommandLineParser::Impl::cat_string(const String& str) const
{
    int left = 0, right = (int)str.length();
    while( left <= right && str[left] == ' ' )
        left++;
    while( right > left && str[right-1] == ' ' )
        right--;
    return left >= right ? String("") : str.substr(left, right-left);
}

String CommandLineParser::getPathToApplication() const
{
    return impl->path_to_app;
}

bool CommandLineParser::has(const String& name) const
{
    for (size_t i = 0; i < impl->data.size(); i++)
    {
        for (size_t j = 0; j < impl->data[i].keys.size(); j++)
        {
            if (name.compare(impl->data[i].keys[j]) == 0 && String("true").compare(impl->data[i].def_value) == 0)
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
        printf("\nERRORS:\n%s\n", impl->error_message.c_str());
        fflush(stdout);
    }
}

void CommandLineParser::printMessage() const
{
    if (impl->about_message != "")
        printf("%s\n", impl->about_message.c_str());

    printf("Usage: %s [params] ", impl->app_name.c_str());

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number > -1)
        {
            String name = impl->data[i].keys[0].substr(1, impl->data[i].keys[0].length() - 1);
            printf("%s ", name.c_str());
        }
    }

    printf("\n\n");

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number == -1)
        {
            printf("\t");
            for (size_t j = 0; j < impl->data[i].keys.size(); j++)
            {
                String k = impl->data[i].keys[j];
                if (k.length() > 1)
                {
                    printf("--");
                }
                else
                {
                    printf("-");
                }
                printf("%s", k.c_str());

                if (j != impl->data[i].keys.size() - 1)
                {
                    printf(", ");
                }
            }
            String dv = impl->cat_string(impl->data[i].def_value);
            if (dv.compare("") != 0)
            {
                printf(" (value:%s)", dv.c_str());
            }
            printf("\n\t\t%s\n", impl->data[i].help_message.c_str());
        }
    }
    printf("\n");

    for (size_t i = 0; i < impl->data.size(); i++)
    {
        if (impl->data[i].number != -1)
        {
            printf("\t");
            String k = impl->data[i].keys[0];
            k = k.substr(1, k.length() - 1);

            printf("%s", k.c_str());

            String dv = impl->cat_string(impl->data[i].def_value);
            if (dv.compare("") != 0)
            {
                printf(" (value:%s)", dv.c_str());
            }
            printf("\n\t\t%s\n", impl->data[i].help_message.c_str());
        }
    }
}

std::vector<String> CommandLineParser::Impl::split_range_string(const String& _str, char fs, char ss) const
{
    String str = _str;
    std::vector<String> vec;
    String word = "";
    bool begin = false;

    while (!str.empty())
    {
        if (str[0] == fs)
        {
            if (begin == true)
            {
                throw cv::Exception(CV_StsParseError,
                         String("error in split_range_string(")
                         + str
                         + String(", ")
                         + String(1, fs)
                         + String(", ")
                         + String(1, ss)
                         + String(")"),
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
                         String("error in split_range_string(")
                         + str
                         + String(", ")
                         + String(1, fs)
                         + String(", ")
                         + String(1, ss)
                         + String(")"),
                         "", __FILE__, __LINE__
                         );
            }
            begin = false;
            vec.push_back(word);
        }

        if (begin == true)
        {
            word = word + str[0];
        }
        str = str.substr(1, str.length() - 1);
    }

    if (begin == true)
    {
        throw cv::Exception(CV_StsParseError,
                 String("error in split_range_string(")
                 + str
                 + String(", ")
                 + String(1, fs)
                 + String(", ")
                 + String(1, ss)
                 + String(")"),
                 "", __FILE__, __LINE__
                );
    }

    return vec;
}

std::vector<String> CommandLineParser::Impl::split_string(const String& _str, char symbol, bool create_empty_item) const
{
    String str = _str;
    std::vector<String> vec;
    String word = "";

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
            word = word + str[0];
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

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include <sstream>

namespace cv
{

namespace {
static const char* noneValue = "<none>";

static String cat_string(const String& str)
{
    int left = 0, right = (int)str.length();
    while( left < right && str[left] == ' ' )
        left++;
    while( right > left && str[right-1] == ' ' )
        right--;
    return left >= right ? String("") : str.substr(left, right-left);
}
}

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

    void apply_params(const String& key, const String& value);
    void apply_params(int i, String value);

    void sort_params();
    int refcount;
};


static const char* get_type_name(int type)
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

// std::tolower is int->int
static char char_tolower(char ch)
{
    return (char)std::tolower((int)ch);
}
static bool parse_bool(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), char_tolower);
    std::istringstream is(str);
    bool b;
    is >> (str.size() > 1 ? std::boolalpha : std::noboolalpha) >> b;
    return b;
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
        *(bool*) dst = parse_bool(temp);
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
    else if( type == Param::SCALAR)
    {
        Scalar& scalar = *(Scalar*)dst;
        for (int i = 0; i < 4 && !ss.eof(); ++i)
            ss >> scalar[i];
    }
    else
        CV_Error(Error::StsBadArg, "unknown/unsupported parameter type");

    if (ss.fail())
    {
        CV_Error_(Error::StsBadArg, ("can not convert: [%s] to [%s]", str.c_str(), get_type_name(type)));
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
                if (name == impl->data[i].keys[j])
                {
                    String v = impl->data[i].def_value;
                    if (space_delete)
                        v = cat_string(v);

                    // the key was neither specified nor has a default value
                    if((v.empty() && type != Param::STRING) || v == noneValue) {
                        impl->error = true;
                        impl->error_message = impl->error_message + "Missing parameter: '" + name + "'\n";
                        return;
                    }

                    from_str(v, type, dst);
                    return;
                }
            }
        }
    }
    catch (const Exception& e)
    {
        impl->error = true;
        impl->error_message = impl->error_message + "Parameter '"+ name + "': " + e.err + "\n";
        return;
    }

    CV_Error_(Error::StsBadArg, ("undeclared key '%s' requested", name.c_str()));
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
                if (space_delete == true) v = cat_string(v);

                // the key was neither specified nor has a default value
                if((v.empty() && type != Param::STRING) || v == noneValue) {
                    impl->error = true;
                    impl->error_message = impl->error_message + format("Missing parameter #%d\n", index);
                    return;
                }
                from_str(v, type, dst);
                return;
            }
        }
    }
    catch (const Exception& e)
    {
        impl->error = true;
        impl->error_message = impl->error_message + format("Parameter #%d: ", index) + e.err + "\n";
        return;
    }

    CV_Error_(Error::StsBadArg, ("undeclared position %d requested", index));
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
        p.help_message = cat_string(l[2]);
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
        String s(argv[i]);
        bool hasSingleDash = s.length() > 1 && s[0] == '-';

        if (hasSingleDash)
        {
            bool hasDoubleDash = s.length() > 2 && s[1] == '-';
            String key = s.substr(hasDoubleDash ? 2 : 1);
            String value = "true";
            size_t equalsPos = key.find('=');

            if(equalsPos != String::npos) {
                value = key.substr(equalsPos + 1);
                key = key.substr(0, equalsPos);
            }
            impl->apply_params(key, value);
        }
        else
        {
            impl->apply_params(jj, s);
            jj++;
        }
    }

    impl->sort_params();
}

CommandLineParser::~CommandLineParser()
{
    if (CV_XADD(&impl->refcount, -1) == 1)
        delete impl;
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
        CV_XADD(&parser.impl->refcount, 1);
        if(CV_XADD(&impl->refcount, -1) == 1)
            delete impl;
        impl = parser.impl;
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
            if (name == impl->data[i].keys[j])
            {
                const String v = cat_string(impl->data[i].def_value);
                return !v.empty() && v != noneValue;
            }
        }
    }

    CV_Error_(Error::StsBadArg, ("undeclared key '%s' requested", name.c_str()));
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
            String dv = cat_string(impl->data[i].def_value);
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

            String dv = cat_string(impl->data[i].def_value);
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

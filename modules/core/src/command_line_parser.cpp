
#include "precomp.hpp"

#include <iostream>

namespace cv
{
    bool cmp_params(const CommandLineParserParams & p1, const CommandLineParserParams & p2)
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

    CommandLineParser::CommandLineParser(int argc, const char * const argv[], const std::string keys)
    {
        // path to application
        size_t pos_s = std::string(argv[0]).find_last_of("/\\");
        if (pos_s == std::string::npos)
        {
            path_to_app = "";
            app_name = std::string(argv[0]);
        }
        else
        {
            path_to_app = std::string(argv[0]).substr(0, pos_s);
            app_name = std::string(argv[0]).substr(pos_s + 1, std::string(argv[0]).length() - pos_s);
        }

        error = false;
        error_message = "";

        // parse keys
        std::vector<std::string> k = split_range_string(keys, '{', '}');

        int jj = 0;
        for (size_t i = 0; i < k.size(); i++)
        {
            std::vector<std::string> l = split_string(k[i], '|', true);
            CommandLineParserParams p;
            p.keys = split_string(l[0]);
            p.def_value = l[1];
            p.help_message = cat_string(l[2]);
            p.number = -1;
            if (p.keys[0][0] == '@')
            {
                p.number = jj;
                jj++;
            }

            data.push_back(p);
        }

        // parse argv
        jj = 0;
        for (int i = 1; i < argc; i++)
        {
            std::string s = std::string(argv[i]);

            if (s.find('=') != std::string::npos && s.find('=') < s.length())
            {
                std::vector<std::string> k_v = split_string(s, '=', true);
                for (int h = 0; h < 2; h++)
                {
                    if (k_v[0][0] == '-')
                        k_v[0] = k_v[0].substr(1, k_v[0].length() -1);
                }
                apply_params(k_v[0], k_v[1]);
            }
            else if (s.length() > 1 && s[0] == '-')
            {
                for (int h = 0; h < 2; h++)
                {
                    if (s[0] == '-')
                        s = s.substr(1, s.length() - 1);
                }
                apply_params(s, "true");
            }
            else if (s[0] != '-')
            {
                apply_params(jj, s);
                jj++;
            }
        }

        sort_params();
    }

    void CommandLineParser::about(std::string message)
    {
        about_message = message;
    }

    void CommandLineParser::apply_params(std::string key, std::string value)
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

    void CommandLineParser::apply_params(int i, std::string value)
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

    void CommandLineParser::sort_params()
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            sort(data[i].keys.begin(), data[i].keys.end());
        }

        sort (data.begin(), data.end(), cmp_params);
    }

    std::string CommandLineParser::cat_string(std::string str)
    {
        while (!str.empty() && str[0] == ' ')
        {
            str = str.substr(1, str.length() - 1);
        }

        while (!str.empty() && str[str.length() - 1] == ' ')
        {
            str = str.substr(0, str.length() - 1);
        }

        return str;
    }

    std::string CommandLineParser::getPathToApplication()
    {
        return path_to_app;
    }

    bool CommandLineParser::has(const std::string& name)
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            for (size_t j = 0; j < data[i].keys.size(); j++)
            {
                if (name.compare(data[i].keys[j]) == 0 && std::string("true").compare(data[i].def_value) == 0)
                {
                    return true;
                }
            }
        }
        return false;
    }

    bool CommandLineParser::check()
    {
        return error == false;
    }

    void CommandLineParser::printErrors()
    {
        if (error)
        {
            std::cout << std::endl << "ERRORS:" << std::endl << error_message << std::endl;
        }
    }

    void CommandLineParser::printMessage()
    {
        if (about_message != "")
            std::cout << about_message << std::endl;

        std::cout << "Usage: " << app_name << " [params] ";

        for (size_t i = 0; i < data.size(); i++)
        {
            if (data[i].number > -1)
            {
                std::string name = data[i].keys[0].substr(1, data[i].keys[0].length() - 1);
                std::cout << name << " ";
            }
        }

        std::cout << std::endl << std::endl;

        for (size_t i = 0; i < data.size(); i++)
        {
            if (data[i].number == -1)
            {
                std::cout << "\t";
                for (size_t j = 0; j < data[i].keys.size(); j++)
                {
                    std::string k = data[i].keys[j];
                    if (k.length() > 1)
                    {
                        std::cout << "--";
                    }
                    else
                    {
                        std::cout << "-";
                    }
                    std::cout << k;

                    if (j != data[i].keys.size() - 1)
                    {
                        std::cout << ", ";
                    }
                }
                std::string dv = cat_string(data[i].def_value);
                if (dv.compare("") != 0)
                {
                    std::cout << " (value:" << dv << ")";
                }
                std::cout << std::endl << "\t\t" << data[i].help_message << std::endl;
            }
        }
        std::cout << std::endl;

        for (size_t i = 0; i < data.size(); i++)
        {
            if (data[i].number != -1)
            {
                std::cout << "\t";
                std::string k = data[i].keys[0];
                k = k.substr(1, k.length() - 1);

                std::cout << k;

                std::string dv = cat_string(data[i].def_value);
                if (dv.compare("") != 0)
                {
                    std::cout << " (value:" << dv << ")";
                }
                std::cout << std::endl << "\t\t" << data[i].help_message << std::endl;
            }
        }
    }

    std::vector<std::string> CommandLineParser::split_range_string(std::string str, char fs, char ss)
    {
        std::vector<std::string> vec;
        std::string word = "";
        bool begin = false;

        while (!str.empty())
        {
            if (str[0] == fs)
            {
                if (begin == true)
                {
                    CV_Error(CV_StsParseError,
                             std::string("error in split_range_string(")
                             + str
                             + std::string(", ")
                             + std::string(1, fs)
                             + std::string(", ")
                             + std::string(1, ss)
                             + std::string(")")
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
                    CV_Error(CV_StsParseError,
                             std::string("error in split_range_string(")
                             + str
                             + std::string(", ")
                             + std::string(1, fs)
                             + std::string(", ")
                             + std::string(1, ss)
                             + std::string(")")
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
            CV_Error(CV_StsParseError,
                     std::string("error in split_range_string(")
                     + str
                     + std::string(", ")
                     + std::string(1, fs)
                     + std::string(", ")
                     + std::string(1, ss)
                     + std::string(")")
                     );
        }

        return vec;
    }

    std::vector<std::string> CommandLineParser::split_string(std::string str, char symbol, bool create_empty_item)
    {
        std::vector<std::string> vec;
        std::string word = "";

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

    #undef clp_get
    #define clp_get(T) template<> T CommandLineParser::get<T>(const std::string& name, bool space_delete);

    clp_get(int)
    clp_get(unsigned int)
    clp_get(long)
    clp_get(unsigned long)
    clp_get(long long)
    clp_get(unsigned long long)
    clp_get(size_t)
    clp_get(float)
    clp_get(double)
    clp_get(uint64)
    clp_get(int64)
    clp_get(std::string)

    #undef clp_from_str
    #define clp_from_str(T) template<> T from_str<T>(const std::string & str);

    clp_from_str(int)
    clp_from_str(unsigned int)
    clp_from_str(long)
    clp_from_str(unsigned long)
    clp_from_str(long long)
    clp_from_str(unsigned long long)
    clp_from_str(size_t)
    clp_from_str(uint64)
    clp_from_str(int64)
    clp_from_str(float)
    clp_from_str(double)

    template<>
    std::string from_str(const std::string & str)
    {
        return str;
    }

    #undef clp_type_name
    #define clp_type_name(type, name) template<> std::string get_type_name<type>() { return std::string(name);}

    clp_type_name(int, "int")
    clp_type_name(unsigned int, "unsigned int")
    clp_type_name(long, "long")
    clp_type_name(long long, "long long")
    clp_type_name(unsigned long long, "unsigned long long")
    clp_type_name(size_t, "size_t")
    clp_type_name(float, "float")
    clp_type_name(double, "double")

}

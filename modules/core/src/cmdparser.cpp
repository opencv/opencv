#include "precomp.hpp"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

void helpParser()
{
    printf("\nThe CommandLineParser class is designed for command line arguments parsing\n"
           "Keys map: \n"
           "Before you start to work with CommandLineParser you have to create a map for keys.\n"
           "    It will look like this\n"
           "    const char* keys =\n"
           "    {\n"
           "        {    s|  string|  123asd |string parameter}\n"
           "        {    d|  digit |  100    |digit parameter }\n"
           "        {    c|noCamera|false    |without camera  }\n"
           "        {    1|        |some text|help            }\n"
           "        {    2|        |333      |another help    }\n"
           "    };\n"
           "Usage syntax: \n"
           "    \"{\" - start of parameter string.\n"
           "    \"}\" - end of parameter string\n"
           "    \"|\" - separator between short name, full name, default value and help\n"
           "Supported syntax: \n"
           "    --key1=arg1  <If a key with '--' must has an argument\n"
           "                  you have to assign it through '=' sign.> \n"
           "<If the key with '--' doesn't have any argument, it means that it is a bool key>\n"
           "    -key2=arg2   <If a key with '-' must has an argument \n"
           "                  you have to assign it through '=' sign.> \n"
           "If the key with '-' doesn't have any argument, it means that it is a bool key\n"
           "    key3                 <This key can't has any parameter> \n"
           "Usage: \n"
           "      Imagine that the input parameters are next:\n"
           "                -s=string_value --digit=250 --noCamera lena.jpg 10000\n"
           "    CommandLineParser parser(argc, argv, keys) - create a parser object\n"
           "    parser.get<string>(\"s\" or \"string\") will return you first parameter value\n"
           "    parser.get<string>(\"s\", false or \"string\", false) will return you first parameter value\n"
           "                                                                without spaces in end and begin\n"
           "    parser.get<int>(\"d\" or \"digit\") will return you second parameter value.\n"
           "                    It also works with 'unsigned int', 'double', and 'float' types>\n"
           "    parser.get<bool>(\"c\" or \"noCamera\") will return you true .\n"
           "                                If you enter this key in commandline>\n"
           "                                It return you false otherwise.\n"
           "    parser.get<string>(\"1\") will return you the first argument without parameter (lena.jpg) \n"
           "    parser.get<int>(\"2\") will return you the second argument without parameter (10000)\n"
           "                          It also works with 'unsigned int', 'double', and 'float' types \n"
           );
}

vector<string> split_string(const string& str, const string& delimiters)
{
	vector<string> res;

	string split_str = str;
	size_t pos_delim = split_str.find(delimiters);

	while ( pos_delim != string::npos)
	{
		if (pos_delim == 0)
		{
			res.push_back("");
			split_str.erase(0, 1);
		}
		else
		{
			res.push_back(split_str.substr(0, pos_delim));
			split_str.erase(0, pos_delim + 1);
		}

		pos_delim = split_str.find(delimiters);
	}

	res.push_back(split_str);

	return res;
}

string del_space(string name)
{
	while ((name.find_first_of(' ') == 0)  && (name.length() > 0))
		name.erase(0, 1);

	while ((name.find_last_of(' ') == (name.length() - 1)) && (name.length() > 0))
		name.erase(name.end() - 1, name.end());

	return name;
}

CommandLineParser::CommandLineParser(int argc, const char* const argv[], const char* keys)
{
	std::string keys_buffer;
    std::string values_buffer;
    std::string buffer;
    std::string curName;
    std::vector<string> keysVector;
    std::vector<string> paramVector;
    std::map<std::string, std::vector<std::string> >::iterator it;
    size_t flagPosition;
    int currentIndex = 1;
    //bool isFound = false;
    bool withNoKey = false;
    bool hasValueThroughEq = false;

    keys_buffer = keys;
    while (!keys_buffer.empty())
    {

        flagPosition = keys_buffer.find_first_of('}');
        flagPosition++;
        buffer = keys_buffer.substr(0, flagPosition);
        keys_buffer.erase(0, flagPosition);

        flagPosition = buffer.find('{');
        if (flagPosition != buffer.npos)
            buffer.erase(flagPosition, (flagPosition + 1));

        flagPosition = buffer.find('}');
        if (flagPosition != buffer.npos)
            buffer.erase(flagPosition);

        paramVector = split_string(buffer, "|");
		while (paramVector.size() < 4) paramVector.push_back("");

		buffer = paramVector[0];
		buffer += '|' + paramVector[1];

		//if (buffer == "") CV_ERROR(CV_StsBadArg, "In CommandLineParser need set short and full name");

		paramVector.erase(paramVector.begin(), paramVector.begin() + 2);
        data[buffer] = paramVector;
    }

    buffer.clear();
    keys_buffer.clear();
    paramVector.clear();
    for (int i = 1; i < argc; i++)
    {
        if (!argv[i])
            break;
        curName = argv[i];
        if (curName.find('-') == 0 && ((curName[1] < '0') || (curName[1] > '9')))
        {
            while (curName.find('-') == 0)
                curName.erase(curName.begin(), (curName.begin() + 1));
        }
            else
                withNoKey = true;
        if (curName.find('=') != curName.npos)
        {
            hasValueThroughEq = true;
            buffer = curName;
            curName.erase(curName.find('='));
            buffer.erase(0, (buffer.find('=') + 1));
        }

		values_buffer = del_space(values_buffer);

        for(it = data.begin(); it != data.end(); it++)
        {
            keys_buffer = it->first;
            keysVector = split_string(keys_buffer, "|");

			for (size_t j = 0; j < keysVector.size(); j++) keysVector[j] = del_space(keysVector[j]);

            values_buffer = it->second[0];
            if (((curName == keysVector[0]) || (curName == keysVector[1])) && hasValueThroughEq)
            {
                it->second[0] = buffer;
                //isFound = true;
                break;
            }

            if (!hasValueThroughEq && ((curName == keysVector[0]) || (curName == keysVector[1]))
				&& (
					values_buffer.find("false") != values_buffer.npos || 
					values_buffer == ""
				))
            {
                it->second[0] = "true";
                //isFound = true;
                break;
            }

            if (!hasValueThroughEq && (values_buffer.find("false") == values_buffer.npos) &&
                ((curName == keysVector[0]) || (curName == keysVector[1])))
            {
                it->second[0] = argv[++i];
                //isFound = true;
                break;
            }


            if (withNoKey)
            {
                std::string noKeyStr = it->first;
                if(atoi(noKeyStr.c_str()) == currentIndex)
                {
                    it->second[0] = curName;
                    currentIndex++;
                    //isFound = true;
                    break;
                }
            }
        }

        withNoKey = false;
        hasValueThroughEq = false;
        //isFound = false;
    }
}

bool CommandLineParser::has(const std::string& keys)
{
    std::map<std::string, std::vector<std::string> >::iterator it;
    std::vector<string> keysVector;

    for(it = data.begin(); it != data.end(); it++)
    {
        keysVector = split_string(it->first, "|");
		for (size_t i = 0; i < keysVector.size(); i++) keysVector[i] = del_space(keysVector[i]);

        if (keysVector.size() == 1) keysVector.push_back("");

		if ((del_space(keys).compare(keysVector[0]) == 0) || 
			(del_space(keys).compare(keysVector[1]) == 0))
            return true;
    }

    return false;
}

std::string CommandLineParser::getString(const std::string& keys)
{
    std::map<std::string, std::vector<std::string> >::iterator it;
    std::vector<string> valueVector;

    for(it = data.begin(); it != data.end(); it++)
    {
        valueVector = split_string(it->first, "|");
		for (size_t i = 0; i < valueVector.size(); i++) valueVector[i] = del_space(valueVector[i]);

        if (valueVector.size() == 1) valueVector.push_back("");

		if ((del_space(keys).compare(valueVector[0]) == 0) || 
			(del_space(keys).compare(valueVector[1]) == 0))
            return it->second[0];
    }
    return string();
}

template<typename _Tp>
 _Tp CommandLineParser::fromStringNumber(const std::string& str)//the default conversion function for numbers
{
    return getData<_Tp>(str);
}

 void CommandLineParser::printParams()
 {
	int col_p = 30;
	int col_d = 50;

	std::map<std::string, std::vector<std::string> >::iterator it;
	std::vector<string> keysVector;
	std::string buf;
	for(it = data.begin(); it != data.end(); it++)
	{
		keysVector = split_string(it->first, "|");
		for (size_t i = 0; i < keysVector.size(); i++) keysVector[i] = del_space(keysVector[i]);

		cout << "  ";
		buf = "";
		if (keysVector[0] != "")
		{
			buf = "-" + keysVector[0];
			if (keysVector[1] != "") buf += ", --" + keysVector[1];
		}
		else if (keysVector[1] != "") buf += "--" + keysVector[1];
		if (del_space(it->second[0]) != "") buf += "=[" + del_space(it->second[0]) + "]";

		cout << setw(col_p-2) << left << buf;

		if ((int)buf.length() > col_p-2) 
		{
			cout << endl << "  ";
			cout << setw(col_p-2) << left << " ";
		}

		buf = "";
		if (del_space(it->second[1]) != "") buf += del_space(it->second[1]);

		for(;;)
		{
			bool tr = ((int)buf.length() > col_d-2) ? true: false;
			std::string::size_type pos = 0;

			if (tr)
			{
				pos = buf.find_first_of(' ');
				for(;;)
				{
					if (buf.find_first_of(' ', pos + 1 ) < (std::string::size_type)(col_d-2) &&
                        buf.find_first_of(' ', pos + 1 ) != std::string::npos)
						pos = buf.find_first_of(' ', pos + 1);
					else
						break;
				}
				pos++;
				cout << setw(col_d-2) << left << buf.substr(0, pos) << endl;
			}
			else
			{
				cout << setw(col_d-2) << left << buf<< endl;
				break;
			}

			buf.erase(0, pos);
			cout << "  ";
			cout << setw(col_p-2) << left << " ";
		}
	}
 }

template<>
bool CommandLineParser::get<bool>(const std::string& name, bool space_delete)
{
    std::string str_buf = getString(name);

	if (space_delete && str_buf != "")
    {
		str_buf = del_space(str_buf);
    }

    if (str_buf == "true")
        return true;

    return false;
}
template<>
std::string CommandLineParser::analyzeValue<std::string>(const std::string& str, bool space_delete)
{
    if (space_delete)
    {
		return del_space(str);
    } 
    return str;
}

template<>
int CommandLineParser::analyzeValue<int>(const std::string& str, bool /*space_delete*/)
{
    return fromStringNumber<int>(str);
}

template<>
unsigned int CommandLineParser::analyzeValue<unsigned int>(const std::string& str, bool /*space_delete*/)
{
    return fromStringNumber<unsigned int>(str);
}

template<>
uint64 CommandLineParser::analyzeValue<uint64>(const std::string& str, bool /*space_delete*/)
{
    return fromStringNumber<uint64>(str);
}

template<>
float CommandLineParser::analyzeValue<float>(const std::string& str, bool /*space_delete*/)
{
    return fromStringNumber<float>(str);
}

template<>
double CommandLineParser::analyzeValue<double>(const std::string& str, bool /*space_delete*/)
{
    return fromStringNumber<double>(str);
}

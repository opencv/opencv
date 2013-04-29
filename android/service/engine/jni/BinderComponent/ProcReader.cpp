#include "ProcReader.h"
#include "StringUtils.h"
#include <fstream>

using namespace std;

map<string, string> GetCpuInfo()
{
    map<string, string> result;
    ifstream f;

    f.open("/proc/cpuinfo");
    if (f.is_open())
    {
        while (!f.eof())
        {
            string tmp;
            string key;
            string value;
            getline(f, tmp);
            if (ParseString(tmp, key, value))
            {
                result[key] = value;
            }
        }
    }

    f.close();

    return result;
}

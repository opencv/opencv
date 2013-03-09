#include "precomp.hpp"

#if defined WIN32 || defined _WIN32 || defined WINCE
# include <windows.h>
const char dir_separators[] = "/\\";
const char native_separator = '\\';
#else
# include <dirent.h>
const char dir_separators[] = "/";
const char native_separator = '/';
#endif

#ifndef _WIN32_WCE
# include <sys/stat.h>
#endif


static bool isDir(std::string path)
{
#ifndef _WIN32_WCE
    struct stat stat_buf;
    stat( path.c_str(), &stat_buf);
    int is_dir = S_ISDIR( stat_buf.st_mode);

    return is_dir != 0;
#else
    /* unsupported */
    return false;
#endif
}

static bool wildcmp(const char *string, const char *wild)
{
    // Based on wildcmp written by Jack Handy - <A href="mailto:jakkhandy@hotmail.com">jakkhandy@hotmail.com</A>
    const char *cp = 0, *mp = 0;

    while ((*string) && (*wild != '*'))
    {
        if ((*wild != *string) && (*wild != '?'))
        {
            return false;
        }

        wild++;
        string++;
    }

    while (*string)
    {
        if (*wild == '*')
        {
            if (!*++wild)
            {
                return true;
            }

            mp = wild;
            cp = string + 1;
        }
        else if ((*wild == *string) || (*wild == '?'))
        {
            wild++;
            string++;
        }
        else
        {
            wild = mp;
            string = cp++;
        }
    }

    while (*wild == '*')
    {
        wild++;
    }

    return *wild == 0;
}

static void glob_rec(const std::string& directory, const std::string& wildchart, std::vector<std::string>& result, bool recursive)
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (directory.c_str())) != 0)
    {
        /* find all the files and directories within directory */
        try
        {
            while ((ent = readdir (dir)) != 0)
            {
                const char* name = ent->d_name;
                if((name[0] == 0) || (name[0] == '.' && name[1] == 0) || (name[0] == '.' && name[1] == '.' && name[2] == 0))
                    continue;

                std::string path = directory + native_separator + name;

                if (isDir(path))
                {
                    if (recursive)
                        glob_rec(path, wildchart, result, recursive);
                }
                else
                {
                    if (wildchart.empty() || wildcmp(name, wildchart.c_str()))
                        result.push_back(path);
                }
            }
        }
        catch (...)
        {
            closedir(dir);
            throw;
        }
        closedir(dir);
    }
    else CV_Error(CV_StsObjectNotFound, "could not open directory");
}

void cv::glob(std::string pattern, std::vector<std::string>& result, bool recursive)
{
    result.clear();
    std::string path, wildchart;

    if (isDir(pattern))
    {
        if(strchr(dir_separators, pattern[pattern.size() - 1]) != 0)
        {
            path = pattern.substr(0, pattern.size() - 1);
        }
        else
        {
            path = pattern;
        }
    }
    else
    {
        size_t pos = pattern.find_last_of(dir_separators);
        if (pos == std::string::npos)
        {
            wildchart = pattern;
            path = ".";
        }
        else
        {
            path = pattern.substr(0, pos);
            wildchart = pattern.substr(pos + 1);
        }
    }

    glob_rec(path, wildchart, result, recursive);
}
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <va/va.h>
# include <va/va_drm.h>

#include "opencv2/core.hpp"  // cv::format()

namespace va {

bool openDisplay();
void closeDisplay();

VADisplay display = NULL;
bool initialized = false;

#define VA_INTEL_PCI_DIR "/sys/bus/pci/devices"
#define VA_INTEL_DRI_DIR "/dev/dri/"
#define VA_INTEL_PCI_DISPLAY_CONTROLLER_CLASS 0x03

static unsigned readId(const char* devName, const char* idName);
static int findAdapter(unsigned desiredVendorId);

int drmfd = -1;

class Directory
{
    typedef int (*fsort)(const struct dirent**, const struct dirent**);
public:
    Directory(const char* path)
        {
            dirEntries = 0;
            numEntries = scandir(path, &dirEntries, filterFunc, (fsort)alphasort);
        }
    ~Directory()
        {
            if (numEntries && dirEntries)
            {
                for (int i = 0;  i < numEntries;  ++i)
                    free(dirEntries[i]);
                free(dirEntries);
            }
        }
    int count() const
        {
            return numEntries;
        }
    const struct dirent* operator[](int index) const
        {
            return ((dirEntries != 0) && (index >= 0) && (index < numEntries)) ? dirEntries[index] : 0;
        }
protected:
    static int filterFunc(const struct dirent* dir)
        {
            if (!dir) return 0;
            if (!strcmp(dir->d_name, ".")) return 0;
            if (!strcmp(dir->d_name, "..")) return 0;
            return 1;
        }
private:
    int numEntries;
    struct dirent** dirEntries;
};

static unsigned readId(const char* devName, const char* idName)
{
    long int id = 0;

    std::string fileName = cv::format("%s/%s/%s", VA_INTEL_PCI_DIR, devName, idName);

    FILE* file = fopen(fileName.c_str(), "r");
    if (file)
    {
        char str[16] = "";
        if (fgets(str, sizeof(str), file))
            id = strtol(str, NULL, 16);
        fclose(file);
    }
    return (unsigned)id;
}

static int findAdapter(unsigned desiredVendorId)
{
    int adapterIndex = -1;

    Directory dir(VA_INTEL_PCI_DIR);

    for (int i = 0;  i < dir.count();  ++i)
    {
        const char* name = dir[i]->d_name;

        unsigned classId = readId(name, "class");
        if ((classId >> 16) == VA_INTEL_PCI_DISPLAY_CONTROLLER_CLASS)
        {
            unsigned vendorId = readId(name, "vendor");
            if (vendorId == desiredVendorId)
            {
                std::string subdirName = cv::format("%s/%s/%s", VA_INTEL_PCI_DIR, name, "drm");
                Directory subdir(subdirName.c_str());
                for (int j = 0; j < subdir.count(); ++j)
                {
                    if (!strncmp(subdir[j]->d_name, "card", 4))
                    {
                        adapterIndex = strtoul(subdir[j]->d_name + 4, NULL, 10);
                    }
                }
                break;
            }
        }
    }

    return adapterIndex;
}

class NodeInfo
{
    enum { NUM_NODES = 2 };
public:
    NodeInfo(int adapterIndex)
        {
            const char* names[NUM_NODES] = { "renderD", "card" };
            int numbers[NUM_NODES];
            numbers[0] = adapterIndex+128;
            numbers[1] = adapterIndex;
            for (int i = 0;  i < NUM_NODES;  ++i)
            {
                paths[i] = cv::format("%s%s%d", VA_INTEL_DRI_DIR, names[i], numbers[i]);
            }
        }
    ~NodeInfo()
        {
            // nothing
        }
    int count() const
        {
            return NUM_NODES;
        }
    const char* path(int index) const
        {
            return ((index >= 0) && (index < NUM_NODES)) ? paths[index].c_str() : 0;
        }
private:
    std::string paths[NUM_NODES];
};

static bool openDeviceIntel();
static bool openDeviceGeneric();

static bool openDeviceIntel()
{
    const unsigned IntelVendorID = 0x8086;

    int adapterIndex = findAdapter(IntelVendorID);
    if (adapterIndex >= 0)
    {
        NodeInfo nodes(adapterIndex);

        for (int i = 0; i < nodes.count(); ++i)
        {
            drmfd = open(nodes.path(i), O_RDWR);
            if (drmfd >= 0)
            {
                display = vaGetDisplayDRM(drmfd);
                if (display)
                    return true;
                close(drmfd);
                drmfd = -1;
            }
        }
    }
    return false;
}

static bool openDeviceGeneric()
{
    static const char* device_paths[] = { "/dev/dri/renderD128", "/dev/dri/card0" };
    static const int num_devices = sizeof(device_paths) / sizeof(device_paths[0]);

    for (int i = 0; i < num_devices; ++i)
    {
        drmfd = open(device_paths[i], O_RDWR);
        if (drmfd >= 0)
        {
            display = vaGetDisplayDRM(drmfd);
            if (display)
                return true;
            close(drmfd);
            drmfd = -1;
        }
    }
    return false;
}

bool openDisplay()
{
    if (!initialized)
    {
        drmfd = -1;
        display = 0;

        if (openDeviceIntel() || openDeviceGeneric())
        {
            int majorVersion = 0, minorVersion = 0;
            if (vaInitialize(display, &majorVersion, &minorVersion) == VA_STATUS_SUCCESS)
            {
                initialized = true;
                return true;
            }
            close(drmfd);
            display = 0;
            drmfd = -1;
        }
        return false; // Can't open VA display
    }
    return true;
}

void closeDisplay()
{
    if (initialized)
    {
        if (display)
            vaTerminate(display);
        if (drmfd >= 0)
            close(drmfd);
        display = 0;
        drmfd = -1;
        initialized = false;
    }
}

} // namespace va

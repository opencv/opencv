#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cvconfig.h"

#include <va/va.h>
#if defined(HAVE_VA_INTEL)
# include <va/va_drm.h>
#elif defined(HAVE_VA)
# include <va/va_x11.h>
# include <X11/Xlib.h>
#endif //HAVE_VA_INTEL / HAVE_VA

namespace va {

#if defined(HAVE_VA_INTEL) || defined(HAVE_VA)

bool openDisplay();
void closeDisplay();

VADisplay display = NULL;
bool initialized = false;

#endif //HAVE_VA_INTEL || HAVE_VA

#if defined(HAVE_VA_INTEL)

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

    char fileName[256];
    snprintf(fileName, sizeof(fileName), "%s/%s/%s", VA_INTEL_PCI_DIR, devName, idName);

    FILE* file = fopen(fileName, "r");
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
    int numAdapters = 0;

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
                adapterIndex = numAdapters;
                break;
            }
            ++numAdapters;
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
                int sz = sizeof(VA_INTEL_DRI_DIR) + strlen(names[i]) + 3;
                paths[i] = new char [sz];
                snprintf(paths[i], sz, "%s%s%d", VA_INTEL_DRI_DIR, names[i], numbers[i]);
            }
        }
    ~NodeInfo()
        {
            for (int i = 0;  i < NUM_NODES;  ++i)
            {
                delete paths[i];
                paths[i] = 0;
            }
        }
    int count() const
        {
            return NUM_NODES;
        }
    const char* path(int index) const
        {
            return ((index >= 0) && (index < NUM_NODES)) ? paths[index] : 0;
        }
private:
    char* paths[NUM_NODES];
};

bool openDisplay()
{
    if (!initialized)
    {
        const unsigned IntelVendorID = 0x8086;

        drmfd = -1;
        display = 0;

        int adapterIndex = findAdapter(IntelVendorID);
        if (adapterIndex >= 0)
        {
            NodeInfo nodes(adapterIndex);

            for (int i = 0;  i < nodes.count();  ++i)
            {
                drmfd = open(nodes.path(i), O_RDWR);
                if (drmfd >= 0)
                {
                    display = vaGetDisplayDRM(drmfd);
                    if (display)
                    {
                        int majorVersion = 0, minorVersion = 0;
                        if (vaInitialize(display, &majorVersion, &minorVersion) == VA_STATUS_SUCCESS)
                        {
                            initialized = true;
                            return true;
                        }
                        display = 0;
                    }
                    close(drmfd);
                    drmfd = -1;
                }
            }
        }

        if (adapterIndex < 0)
            return false; // Can't find Intel display adapter
        if ((drmfd < 0) || !display)
            return false; // Can't load VA display
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

#elif defined(HAVE_VA)

static Display* x11Display = 0;

bool openDisplay()
{
    if (!initialized)
    {
        display = 0;

        x11Display = XOpenDisplay("");
        if (x11Display != 0)
        {
            display = vaGetDisplay(x11Display);
            if (display)
            {
                int majorVersion = 0, minorVersion = 0;
                if (vaInitialize(display, &majorVersion, &minorVersion) == VA_STATUS_SUCCESS)
                {
                    initialized = true;
                    return true;
                }
                display = 0;
            }
            XCloseDisplay(x11Display);
            x11Display = 0;
        }

        return false; // Can't initialize X11/VA display
    }
    return true;
}

void closeDisplay()
{
    if (initialized)
    {
        if (display)
            vaTerminate(display);
        if (x11Display)
            XCloseDisplay(x11Display);
        display = 0;
        x11Display = 0;
        initialized = false;
    }
}

#endif // HAVE_VA_INTEL / HAVE_VA

} // namespace va

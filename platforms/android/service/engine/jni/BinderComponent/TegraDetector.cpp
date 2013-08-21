#include "TegraDetector.h"
#include <zlib.h>
#include <string.h>

#define KERNEL_CONFIG "/proc/config.gz"
#define KERNEL_CONFIG_MAX_LINE_WIDTH 512
#define KERNEL_CONFIG_TEGRA_MAGIC "CONFIG_ARCH_TEGRA=y"
#define KERNEL_CONFIG_TEGRA2_MAGIC "CONFIG_ARCH_TEGRA_2x_SOC=y"
#define KERNEL_CONFIG_TEGRA3_MAGIC "CONFIG_ARCH_TEGRA_3x_SOC=y"
#define KERNEL_CONFIG_TEGRA4_MAGIC "CONFIG_ARCH_TEGRA_11x_SOC=y"
#define MAX_DATA_LEN    4096

int DetectTegra()
{
    int result = TEGRA_NOT_TEGRA;
    gzFile kernelConfig = gzopen(KERNEL_CONFIG, "r");
    if (kernelConfig != 0)
    {
        char tmpbuf[KERNEL_CONFIG_MAX_LINE_WIDTH];
        const char *tegra_config = KERNEL_CONFIG_TEGRA_MAGIC;
        const char *tegra2_config = KERNEL_CONFIG_TEGRA2_MAGIC;
        const char *tegra3_config = KERNEL_CONFIG_TEGRA3_MAGIC;
        const char *tegra4_config = KERNEL_CONFIG_TEGRA4_MAGIC;
        int len = strlen(tegra_config);
        int len2 = strlen(tegra2_config);
        int len3 = strlen(tegra3_config);
        int len4 = strlen(tegra4_config);
        while (0 != gzgets(kernelConfig, tmpbuf, KERNEL_CONFIG_MAX_LINE_WIDTH))
        {
            if (0 == strncmp(tmpbuf, tegra_config, len))
            {
                result = 1;
            }

            if (0 == strncmp(tmpbuf, tegra2_config, len2))
            {
                result = 2;
                break;
            }

            if (0 == strncmp(tmpbuf, tegra3_config, len3))
            {
                result = 3;
                break;
            }

            if (0 == strncmp(tmpbuf, tegra4_config, len4))
            {
                result = 4;
                break;
            }
        }
        gzclose(kernelConfig);
    }
    else
    {
        result = TEGRA_DETECTOR_ERROR;
    }

    return result;
}

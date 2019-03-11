package org.opencv.engine;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import android.os.Build;
import android.text.TextUtils;
import android.util.Log;

public class HardwareDetector {
    private static String TAG = "OpenCVEngine/HardwareDetector";

    public static final int ARCH_UNKNOWN = -1;

    public static final int ARCH_X86 = 0x01000000;
    public static final int ARCH_X86_64 = 0x02000000;
    public static final int ARCH_ARM = 0x04000000;
    public static final int ARCH_ARMv7 = 0x10000000;
    public static final int ARCH_ARMv8 = 0x20000000;
    public static final int ARCH_MIPS = 0x40000000;
    public static final int ARCH_MIPS_64 = 0x80000000;

    // Return CPU flags list
    public static List<String> getFlags() {
        Map<String, String> raw = getRawCpuInfo();
        String f = raw.get("flags");
        if (f == null)
            f = raw.get("Features");
        if (f == null)
            return Arrays.asList();
        return Arrays.asList(TextUtils.split(f, " "));
    }

    // Return CPU arch
    public static int getAbi() {
        List<String> abis = Arrays.asList(Build.CPU_ABI, Build.CPU_ABI2);
        Log.i(TAG, "ABIs: " + abis.toString());
        if (abis.contains("x86_64")) {
            return ARCH_X86_64;
        } else if (abis.contains("x86")) {
            return ARCH_X86;
        } else if (abis.contains("arm64-v8a")) {
            return ARCH_ARMv8;
        } else if (abis.contains("armeabi-v7a")
                || abis.contains("armeabi-v7a-hard")) {
            return ARCH_ARMv7;
        } else if (abis.contains("armeabi")) {
            return ARCH_ARM;
        } else if (abis.contains("mips64")) {
            return ARCH_MIPS_64;
        } else if (abis.contains("mips")) {
            return ARCH_MIPS;
        }
        return ARCH_UNKNOWN;
    }

    // Return hardware platform name
    public static String getHardware() {
        Map<String, String> raw = getRawCpuInfo();
        return raw.get("Hardware");
    }

    // Return processor count
    public static int getProcessorCount() {
        int result = 0;
        try {
            Pattern pattern = Pattern.compile("(\\d)+(-(\\d+))?");
            Scanner s = new Scanner(
                    new File("/sys/devices/system/cpu/possible"));
            if (s.hasNextLine()) {
                String line = s.nextLine();
                Log.d(TAG, "Got CPUs: " + line);
                Matcher m = pattern.matcher(line);
                while (m.find()) {
                    int start = Integer.parseInt(m.group(1));
                    int finish = start;
                    if (m.group(3) != null) {
                        finish = Integer.parseInt(m.group(3));
                    }
                    result += finish - start + 1;
                    Log.d(TAG, "Got CPU range " + start + " ~ " + finish);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to read cpu count");
            e.printStackTrace();
        }
        return result;

    }

    // Return parsed cpuinfo contents
    public static Map<String, String> getRawCpuInfo() {
        Map<String, String> map = new HashMap<String, String>();
        try {
            Scanner s = new Scanner(new File("/proc/cpuinfo"));
            while (s.hasNextLine()) {
                String line = s.nextLine();
                String[] vals = line.split(": ");
                if (vals.length > 1) {
                    map.put(vals[0].trim(), vals[1].trim());
                } else {
                    Log.d(TAG, "Failed to parse cpuinfo: " + line);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to read cpuinfo");
            e.printStackTrace();
        }
        return map;
    }

}

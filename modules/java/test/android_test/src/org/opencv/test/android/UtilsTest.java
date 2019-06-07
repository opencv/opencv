package org.opencv.test.android;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import android.util.Log;

import junit.framework.TestCase;

public class UtilsTest extends TestCase {

    public void testDebug() throws IOException {
        File file = new File(getClass().getProtectionDomain().getCodeSource().getLocation().getFile());
        System.out.println("Initial location " + file);
        Log.e("Bmp->Mat", "Initial location " + file);
        File build = getBuild(file);
        System.out.println("Build " + build);
        if (build != null) {
            print(build.getParentFile());
            printContent(build);
        }
    }

    private static File getBuild(File f) {
        File parent = f;
        do {
            parent = parent.getParentFile();
            File build = new File(parent, "build.xml");
            if (build.exists()) {
                return build;
            }
        } while (parent != null);
        return null;
    }

    private static void print(File f) {
        for (File child : f.listFiles()) {
            System.out.println(child);
            Log.e("Bmp->Mat", child.toString());
            if (child.isDirectory()) {
                print(child);
            }
        }
    }
    private static void printContent(File f) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(f));
        String line;
        while ((line = reader.readLine()) != null) {
            Log.e("Bmp->Mat", line);
            System.out.println(line);
        }
        reader.close();
    }
}

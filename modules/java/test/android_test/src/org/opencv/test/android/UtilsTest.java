package org.opencv.test.android;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import junit.framework.TestCase;

public class UtilsTest extends TestCase {

    public void testDebug() throws IOException {
        StringBuilder builder = new StringBuilder();
        File file = new File(getClass().getProtectionDomain().getCodeSource().getLocation().getFile());
        System.out.println("Initial location " + file);
        builder.append("Initial location " + file).append("\n");

        File build = getBuild(file);
        System.out.println("Build " + build);
        System.err.println("Build " + build);
        builder.append("Build " + build).append("\n");
        if (build != null) {
            print(builder, build.getParentFile());
            printContent(builder, build);
        }
        assertEquals("", builder.toString());
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

    private static void print(StringBuilder builder, File f) {
        for (File child : f.listFiles()) {
            System.out.println(child);
            System.err.println(child);
            builder.append(child).append("\n");
            if (child.isDirectory()) {
                print(builder, child);
            }
        }
    }
    private static void printContent(StringBuilder builder, File f) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(f));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
            System.err.println(line);
            builder.append(line).append("\n");
        }
        reader.close();
    }
}

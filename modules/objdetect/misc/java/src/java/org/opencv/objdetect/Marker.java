package org.opencv.objdetect;
public class Marker extends Aruco2_Marker {
    protected Marker(long addr) { super(addr); }
    public static Marker __fromPtr__(long addr) { return new Marker(addr); }
}

package org.opencv.objdetect;
public class DetectionParameters extends Aruco2_DetectionParameters {
    protected DetectionParameters(long addr) { super(addr); }
    public static DetectionParameters __fromPtr__(long addr) { return new DetectionParameters(addr); }
    public DetectionParameters() { super(Aruco2_DetectionParameters_0()); }
}

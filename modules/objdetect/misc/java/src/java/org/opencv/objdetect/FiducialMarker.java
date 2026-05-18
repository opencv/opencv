package org.opencv.objdetect;
public class FiducialMarker extends Aruco2_FiducialMarker {
    protected FiducialMarker(long addr) { super(addr); }
    public static FiducialMarker __fromPtr__(long addr) { return new FiducialMarker(addr); }
    public FiducialMarker() { super(Objdetect.Aruco2_FiducialMarker_0()); }
}

package org.opencv.objdetect;
public class FractalMarker extends Aruco2_FractalMarker {
    protected FractalMarker(long addr) { super(addr); }
    public static FractalMarker __fromPtr__(long addr) { return new FractalMarker(addr); }
    public FractalMarker() { super(Objdetect.Aruco2_FractalMarker_0()); }
}

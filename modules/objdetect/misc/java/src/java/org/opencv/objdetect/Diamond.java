package org.opencv.objdetect;
public class Diamond extends Aruco2_Diamond {
    protected Diamond(long addr) { super(addr); }
    public static Diamond __fromPtr__(long addr) { return new Diamond(addr); }
    public Diamond() { super(Objdetect.Aruco2_Diamond_0()); }
}

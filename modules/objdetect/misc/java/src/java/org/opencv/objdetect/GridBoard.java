package org.opencv.objdetect;
public class GridBoard extends Aruco2_GridBoard {
    protected GridBoard(long addr) { super(addr); }
    public static GridBoard __fromPtr__(long addr) { return new GridBoard(addr); }
}

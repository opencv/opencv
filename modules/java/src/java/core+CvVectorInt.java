package org.opencv.core;


public class CvVectorInt extends CvVector {
	private static final int _d = CvType.CV_32S;
	
    public CvVectorInt(int ch) {
        super(_d, ch);
    }

    public CvVectorInt(int ch, long addr) {
        super(_d, ch, addr);
    }

    public CvVectorInt(int ch, Mat m) {
        super(_d, ch, m);
    }

    public CvVectorInt(int ch, int[] a) {
        super(_d, ch);
        if(a!=null) {
        	int cnt = a.length / ch;
        	create(cnt);
        	put(0, 0, a);
        }
    }

    public int[] toArray(int[] a) {
        int cnt = (int) total() * channels;
        if(cnt == 0)
            return new int[0];//null;
        int[] res = a;
        if(res==null || res.length<cnt)
            res = new int[cnt];
        get(0, 0, res); //TODO: check ret val!
        return res;
    }
}

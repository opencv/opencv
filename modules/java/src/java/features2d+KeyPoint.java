package org.opencv.features2d;

import org.opencv.core.Point;

//javadoc: KeyPoint
public class KeyPoint {
	
    //javadoc: KeyPoint::pt
    public Point pt;
    //javadoc: KeyPoint::size
    public float size;
    //javadoc: KeyPoint::angle
    public float angle;
    //javadoc: KeyPoint::response
    public float response;
    //javadoc: KeyPoint::octave
    public int octave;
    //javadoc: KeyPoint::class_id
    public int class_id; 

    //javadoc: KeyPoint::KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)
    public KeyPoint(float x, float y, float _size, float _angle, float _response, int _octave, int _class_id)
    {
    	 pt       = new Point(x, y);
    	 size     = _size;
    	 angle    = _angle;
    	 response = _response;
    	 octave   = _octave;
    	 class_id = _class_id;
    }

    //javadoc: KeyPoint::KeyPoint()
    public KeyPoint()
    {
    	this(0, 0, 0, -1, 0, 0, -1);
    }

    //javadoc: KeyPoint::KeyPoint(x, y, _size, _angle, _response, _octave)
    public KeyPoint(float x, float y, float _size, float _angle, float _response, int _octave)
    {
    	this(x, y, _size, _angle, _response, _octave, -1);
    }

    //javadoc: KeyPoint::KeyPoint(x, y, _size, _angle, _response)
    public KeyPoint(float x, float y, float _size, float _angle, float _response)
    {
    	this(x, y, _size, _angle, _response, 0, -1);
    }

    //javadoc: KeyPoint::KeyPoint(x, y, _size, _angle)
    public KeyPoint(float x, float y, float _size, float _angle)
    {
    	this(x, y, _size, _angle, 0, 0, -1);
    }

    //javadoc: KeyPoint::KeyPoint(x, y, _size)
    public KeyPoint(float x, float y, float _size)
    {
    	this(x, y, _size, -1, 0, 0, -1);
    }

	@Override
	public String toString() {
		return "KeyPoint [pt=" + pt + ", size=" + size + ", angle=" + angle
				+ ", response=" + response + ", octave=" + octave
				+ ", class_id=" + class_id + "]";
	}

}

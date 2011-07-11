package org.opencv;

public class CvException extends Exception {

	private static final long serialVersionUID = 1L;

	public CvException(String msg) {
		super(msg);
	}

	@Override
	public String toString() {
		return "CvException [" + super.toString() + "]";
	}
}

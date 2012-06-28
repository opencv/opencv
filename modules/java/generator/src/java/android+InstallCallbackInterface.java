package org.opencv.android;

/**
 * Installation callback interface
 */
public interface InstallCallbackInterface
{
    /**
     * Target package name
     * @return Return target package name
     */
    public String getPackageName();
    /**
     * Installation of package is approved
     */
    public void install();
    /**
     * Installation canceled
     */
    public void cancel();
};

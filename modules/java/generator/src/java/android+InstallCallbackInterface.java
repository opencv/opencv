package org.opencv.android;

/**
 * Installation callback interface.
 */
public interface InstallCallbackInterface
{
    /**
     * Target package name.
     * @return Return target package name.
     */
    public String getPackageName();
    /**
     * Installation is approved.
     */
    public void install();
    /**
     * Installation is canceled.
     */
    public void cancel();
};

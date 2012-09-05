package org.opencv.android;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.DialogInterface.OnClickListener;
import android.util.Log;

/**
 * Basic implementation of LoaderCallbackInterface.
 */
public abstract class BaseLoaderCallback implements LoaderCallbackInterface {

    public BaseLoaderCallback(Activity AppContext) {
        mAppContext = AppContext;
    }

    public void onManagerConnected(int status)
    {
        switch (status)
        {
            /** OpenCV initialization was successful. **/
            case LoaderCallbackInterface.SUCCESS:
            {
                /** Application must override this method to handle successful library initialization. **/
            } break;
            /** OpenCV Manager or library package installation is in progress. Restart the application. **/
            case LoaderCallbackInterface.RESTART_REQUIRED:
            {
                Log.d(TAG, "OpenCV downloading. App restart is needed!");
                AlertDialog RestartMessage = new AlertDialog.Builder(mAppContext).create();
                RestartMessage.setTitle("App restart is required");
                RestartMessage.setMessage("Application will be closed now. Start it when installation will be finished!");
                RestartMessage.setCancelable(false); // This blocks the 'BACK' button
                RestartMessage.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });

                RestartMessage.show();
            } break;
            /** OpenCV loader can not start Google Play Market. **/
            case LoaderCallbackInterface.MARKET_ERROR:
            {
                Log.d(TAG, "Google Play service is not installed! You can get it here");
                AlertDialog MarketErrorMessage = new AlertDialog.Builder(mAppContext).create();
                MarketErrorMessage.setTitle("OpenCV Manager");
                MarketErrorMessage.setMessage("Package installation failed!");
                MarketErrorMessage.setCancelable(false); // This blocks the 'BACK' button
                MarketErrorMessage.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });
                MarketErrorMessage.show();
            } break;
            /** Package installation has been canceled. **/
            case LoaderCallbackInterface.INSTALL_CANCELED:
            {
                Log.d(TAG, "OpenCV library instalation was canceled by user");
                mAppContext.finish();
            } break;
            /** Application is incompatible with this version of OpenCV Manager. Possibly, a service update is required. **/
            case LoaderCallbackInterface.INCOMPATIBLE_MANAGER_VERSION:
            {
                Log.d(TAG, "OpenCV Manager Service is uncompatible with this app!");
                AlertDialog IncomatibilityMessage = new AlertDialog.Builder(mAppContext).create();
                IncomatibilityMessage.setTitle("OpenCV Manager");
                IncomatibilityMessage.setMessage("OpenCV Manager service is incompatible with this app. Update it!");
                IncomatibilityMessage.setCancelable(false); // This blocks the 'BACK' button
                IncomatibilityMessage.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });
                IncomatibilityMessage.show();
            }
            /** Other status, i.e. INIT_FAILED. **/
            default:
            {
                Log.e(TAG, "OpenCV loading failed!");
                AlertDialog InitFailedDialog = new AlertDialog.Builder(mAppContext).create();
                InitFailedDialog.setTitle("OpenCV error");
                InitFailedDialog.setMessage("OpenCV was not initialised correctly. Application will be shut down");
                InitFailedDialog.setCancelable(false); // This blocks the 'BACK' button
                InitFailedDialog.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new OnClickListener() {

                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });

                InitFailedDialog.show();
            } break;
        }
    }

    public void onPackageInstall(final InstallCallbackInterface callback)
    {
        AlertDialog InstallMessage = new AlertDialog.Builder(mAppContext).create();
        InstallMessage.setTitle("Package not found");
        InstallMessage.setMessage(callback.getPackageName() + " package was not found! Try to install it?");
        InstallMessage.setCancelable(false); // This blocks the 'BACK' button
        InstallMessage.setButton(AlertDialog.BUTTON_POSITIVE, "Yes", new OnClickListener()
        {
            public void onClick(DialogInterface dialog, int which)
            {
                callback.install();
            }
        });

        InstallMessage.setButton(AlertDialog.BUTTON_NEGATIVE, "No", new OnClickListener() {

            public void onClick(DialogInterface dialog, int which)
            {
                callback.cancel();
            }
        });

        InstallMessage.show();
    }

    protected Activity mAppContext;
    private final static String TAG = "OpenCVLoader/BaseLoaderCallback";
}

package org.opencv.android;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.DialogInterface.OnClickListener;
import android.util.Log;

/**
 * Basic implementation of LoaderCallbackInterface
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
                /** Application must override this method to handle successful library initialization **/
            } break;
            /** OpenCV Manager or library package installation is in progress. Restart of application is required **/
            case LoaderCallbackInterface.RESTART_REQUIRED:
            {
                Log.d(TAG, "OpenCV downloading. App restart is needed!");
                AlertDialog RestartMessage = new AlertDialog.Builder(mAppContext).create();
                RestartMessage.setTitle("OpenCV asks for restart");
                RestartMessage.setMessage("Please, restart the application to finish installation.");
                RestartMessage.setCancelable(false); // This blocks the 'BACK' button
                RestartMessage.setButton("OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });

                RestartMessage.show();
            } break;
            /** OpenCV loader cannot start Google Play **/
            case LoaderCallbackInterface.MARKET_ERROR:
            {
                Log.d(TAG, "Google Play service is not installed! You can get it here");
                AlertDialog MarketErrorMessage = new AlertDialog.Builder(mAppContext).create();
                MarketErrorMessage.setTitle("OpenCV Manager");
                MarketErrorMessage.setMessage("Package installation failed!");
                MarketErrorMessage.setCancelable(false); // This blocks the 'BACK' button
                MarketErrorMessage.setButton("OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });
                MarketErrorMessage.show();
            } break;
            /** Package installation was canceled **/
            case LoaderCallbackInterface.INSTALL_CANCELED:
            {
                Log.d(TAG, "OpenCV library instalation was canceled by user");
                mAppContext.finish();
            } break;
            /** Application is incompatible with this version of OpenCV Manager. Possible Service update is needed **/
            case LoaderCallbackInterface.INCOMPATIBLE_MANAGER_VERSION:
            {
                Log.d(TAG, "OpenCV Manager Service is uncompatible with this app!");
                AlertDialog IncomatibilityMessage = new AlertDialog.Builder(mAppContext).create();
                IncomatibilityMessage.setTitle("OpenCV Manager");
                IncomatibilityMessage.setMessage("OpenCV Manager service is incompatible with this application. Update it!");
                IncomatibilityMessage.setCancelable(false); // This blocks the 'BACK' button
                IncomatibilityMessage.setButton("OK", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        mAppContext.finish();
                    }
                });
                IncomatibilityMessage.show();
            }
            /** Other status, i.e. INIT_FAILED **/
            default:
            {
                Log.e(TAG, "OpenCV loading failed!");
                AlertDialog InitFailedDialog = new AlertDialog.Builder(mAppContext).create();
                InitFailedDialog.setTitle("OpenCV error");
                InitFailedDialog.setMessage("OpenCV was not initialised correctly. Application will be shut down");
                InitFailedDialog.setCancelable(false); // This blocks the 'BACK' button
                InitFailedDialog.setButton("OK", new OnClickListener() {

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
        InstallMessage.setButton("Yes", new OnClickListener()
        {
            public void onClick(DialogInterface dialog, int which)
            {
                callback.install();
            }
        });

        InstallMessage.setButton2("No", new OnClickListener() {

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

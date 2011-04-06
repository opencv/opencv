package com.opencv.calibration.services;

import java.io.File;
import java.io.IOException;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;


import com.opencv.R;
import com.opencv.calibration.CalibrationViewer;
import com.opencv.calibration.Calibrator;
import com.opencv.calibration.Calibrator.CalibrationCallback;


public class CalibrationService extends Service implements CalibrationCallback {

	Class<?> activity;
	int icon;
	File calibration_file;
	public void startCalibrating(Class<?> activitycaller,int icon_id, Calibrator calibrator, File calibration_file)
			throws IOException {
		activity = activitycaller;
		icon = icon_id;
		// Display a notification about us starting. We put an icon in the
		// status bar.
		showNotification();
		this.calibration_file = calibration_file;
		calibrator.setCallback(this);
		calibrator.calibrate(calibration_file);
		
		
	}

	private NotificationManager mNM;

	/**
	 * Class for clients to access. Because we know this service always runs in
	 * the same process as its clients, we don't need to deal with IPC.
	 */
	public class CalibrationServiceBinder extends Binder {
		public CalibrationService getService() {
			return CalibrationService.this;
		}
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		Log.i("LocalService", "Received start id " + startId + ": " + intent);
		// We want this service to continue running until it is explicitly
		// stopped, so return sticky.
		return START_NOT_STICKY;
	}

	@Override
	public void onCreate() {
		mNM = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);

		
	}

	@Override
	public void onDestroy() {
		// Cancel the persistent notification.
		// mNM.cancel(R.string.calibration_service_started);

		// Tell the user we stopped.
		Toast.makeText(this, R.string.calibration_service_finished,
				Toast.LENGTH_SHORT).show();
	}

	private final IBinder mBinder = new CalibrationServiceBinder();

	@Override
	public IBinder onBind(Intent intent) {
		return mBinder;
	}

	/**
	 * Show a notification while this service is running.
	 */
	private void showNotification() {
		// In this sample, we'll use the same text for the ticker and the
		// expanded notification
		CharSequence text = getText(R.string.calibration_service_started);

		// Set the icon, scrolling text and timestamp
		Notification notification = new Notification(icon, text,
				System.currentTimeMillis());

		// The PendingIntent to launch our activity if the user selects this
		// notification
		PendingIntent contentIntent = PendingIntent.getActivity(this, 0,
				new Intent(this, activity), 0);

		// Set the info for the views that show in the notification panel.
		notification.setLatestEventInfo(this,
				getText(R.string.calibration_service_label), text,
				contentIntent);

		notification.defaults |= Notification.DEFAULT_SOUND;
		// Send the notification.
		// We use a layout id because it is a unique number. We use it later to
		// cancel.
		mNM.notify(R.string.calibration_service_started, notification);
	}

	/**
	 * Show a notification while this service is running.
	 */
	private void doneNotification() {
		// In this sample, we'll use the same text for the ticker and the
		// expanded notification
		CharSequence text = getText(R.string.calibration_service_finished);

		// Set the icon, scrolling text and timestamp
		Notification notification = new Notification(icon, text,
				System.currentTimeMillis());

		Intent intent = new Intent(this,CalibrationViewer.class);
		intent.putExtra("calibfile", calibration_file.getAbsolutePath());
		// The PendingIntent to launch our activity if the user selects this
		// notification
		PendingIntent contentIntent = PendingIntent.getActivity(this, 0,
				intent, 0);
		

		// Set the info for the views that show in the notification panel.
		notification.setLatestEventInfo(this,
				getText(R.string.calibration_service_label), text,
				contentIntent);
		

		notification.defaults |= Notification.DEFAULT_SOUND;
		// Send the notification.
		// We use a layout id because it is a unique number. We use it later to
		// cancel.
		mNM.notify(R.string.calibration_service_started, notification);
	}

	@Override
	public void onFoundChessboard(Calibrator calibrator) {
		// TODO Auto-generated method stub

	}

	@Override
	public void onDoneCalibration(Calibrator calibration, File calibfile) {
		doneNotification();
		stopSelf();
	}

	@Override
	public void onFailedChessboard(Calibrator calibrator) {
		// TODO Auto-generated method stub

	}

}

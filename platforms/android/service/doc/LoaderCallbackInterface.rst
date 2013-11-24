*************************
Loader Callback Interface
*************************

.. highlight:: java
.. class:: LoaderCallbackInterface

    Interface for a callback object in case of asynchronous initialization of OpenCV.

void onManagerConnected()
-------------------------

.. method:: void onManagerConnected(int status)

    Callback method that is called after OpenCV library initialization.

    :param status: status of initialization (see "Initialization Status Constants" section below).

void onPackageInstall()
-----------------------

.. method:: void onPackageInstall(InstallCallbackInterface Callback)

    Callback method that is called in case when package installation is needed.

    :param callback: answer object with ``install`` and ``cancel`` methods and package description.

Initialization status constants
-------------------------------

.. data:: SUCCESS

    OpenCV initialization finished successfully

.. data:: MARKET_ERROR

    Google Play (Android Market) application cannot be invoked

.. data:: INSTALL_CANCELED

    OpenCV library installation was cancelled by user

.. data:: INCOMPATIBLE_MANAGER_VERSION

    Version of OpenCV Manager is incompatible with this app. Manager update is needed.

.. data:: INIT_FAILED

    OpenCV library initialization failed

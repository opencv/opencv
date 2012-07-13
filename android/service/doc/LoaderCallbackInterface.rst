*************************
Loader Callback Interface
*************************

.. highlight:: java
.. class:: LoaderCallbackInterface

    Interface for callback object in case of asynchronous initialization of OpenCV

void onManagerConnected()
-------------------------

.. method:: void onManagerConnected(int status)

    Callback method that is called after OpenCV library initialization

    :param status: Status of initialization. See Initialization status constants

void onPackageInstall()
-----------------------

.. method:: void onPackageInstall(InstallCallbackInterface Callback)

    Callback method that is called in case when package installation is needed

    :param callback: Answer object with approve and cancel methods and package description

Initialization status constants
-------------------------------

.. data:: SUCCESS

    OpenCV initialization finished successfully

.. data:: RESTART_REQUIRED

    OpenCV library installation via Google Play service was initialized. Application restart is required

.. data:: MARKET_ERROR

    Google Play (Android Market) cannot be invoked

.. data:: INSTALL_CANCELED

    OpenCV library installation was canceled by user

.. data:: INCOMPATIBLE_MANAGER_VERSION

    Version of OpenCV Manager Service is incompatible with this app. Service update is needed

.. data:: INIT_FAILED

    OpenCV library initialization failed
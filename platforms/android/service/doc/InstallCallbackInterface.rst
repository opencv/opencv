**************************
Install Callback Interface
**************************
.. highlight:: java
.. class:: InstallCallbackInterface

    Callback interface for package installation or update.

String getPackageName()
-----------------------

.. method:: String getPackageName()

    Get name of a package to be installed.

    :rtype: string;
    :return: returns package name, i.e. "OpenCV Manager Service" or "OpenCV library".

void install()
--------------

.. method:: void install()

    Installation of package has been approved.

void cancel()
-------------

.. method:: void cancel()

    Installation of package has been cancelled.

void wait_install()
-------------------

.. method:: void wait_install()

    Wait for package installation.

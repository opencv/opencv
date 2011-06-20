*************************************
highgui. High-level GUI and Media I/O
*************************************


While OpenCV was designed for use in full-scale
applications and can be used within functionally rich UI frameworks (such as Qt, WinForms or Cocoa) or without any UI at all, sometimes there is a need to try some functionality quickly and visualize the results. This is what the HighGUI module has been designed for.

It provides easy interface to:


    

*
    create and manipulate windows that can display images and "remember" their content (no need to handle repaint events from OS)
        
    

*
    add trackbars to the windows, handle simple mouse events as well as keyboard commmands
        
    

*
    read and write images to/from disk or memory.
        
    

*
    read video from camera or file and write video to a file.
    
    

.. toctree::
    :maxdepth: 2

    highgui_user_interface
    highgui_reading_and_writing_images_and_video

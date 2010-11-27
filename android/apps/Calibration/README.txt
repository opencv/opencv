to build completeley from command line:
sh project_create.sh
ant debug
ant install

That assumes that you have already build the opencv/android/android-jni project

If you're in eclipse, try to create a new android project from existing sources.
Make sure that you also have the android-jni project open in eclipse is this is the case
or the android library dependency will give you errors.
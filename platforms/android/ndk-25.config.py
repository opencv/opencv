# Docs: https://developer.android.com/ndk/guides/cmake#android_native_api_level
ANDROID_NATIVE_API_LEVEL = int(os.environ.get('ANDROID_NATIVE_API_LEVEL', 32))
cmake_common_vars = {
    # Docs: https://source.android.com/docs/setup/about/build-numbers
    # Docs: https://developer.android.com/studio/publish/versioning
    'ANDROID_COMPILE_SDK_VERSION': os.environ.get('ANDROID_COMPILE_SDK_VERSION', 32),
    'ANDROID_TARGET_SDK_VERSION': os.environ.get('ANDROID_TARGET_SDK_VERSION', 32),
    'ANDROID_MIN_SDK_VERSION': os.environ.get('ANDROID_MIN_SDK_VERSION', ANDROID_NATIVE_API_LEVEL),
    # Docs: https://developer.android.com/studio/releases/gradle-plugin
    'ANDROID_GRADLE_PLUGIN_VERSION': '7.3.1',
    'GRADLE_VERSION': '7.5.1',
    'KOTLIN_PLUGIN_VERSION': '1.8.20',
}
ABIs = [
    ABI("2", "armeabi-v7a", None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
    ABI("3", "arm64-v8a",   None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
    ABI("5", "x86_64",      None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
    ABI("4", "x86",         None, ndk_api_level=ANDROID_NATIVE_API_LEVEL, cmake_vars=cmake_common_vars),
]

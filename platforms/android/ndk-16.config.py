ABIs = [
    ABI("2", "armeabi-v7a", "arm-linux-androideabi-4.9", cmake_vars=dict(ANDROID_ABI='armeabi-v7a with NEON')),
    ABI("1", "armeabi",     "arm-linux-androideabi-4.9", cmake_vars=dict(WITH_TBB='OFF')),
    ABI("3", "arm64-v8a",   "aarch64-linux-android-4.9"),
    ABI("5", "x86_64",      "x86_64-4.9"),
    ABI("4", "x86",         "x86-4.9"),
]

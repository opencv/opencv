sysroot_bin_dir = prepare_dir(self.sysrootdir / 'bin')
copytree(self.build_dir / 'install', self.sysrootdir / 'ngraph')
#rm_one(self.sysrootdir / 'ngraph' / 'lib' / 'ngraph.dll')

build_config = 'Release' if not self.config.build_debug else 'Debug'
build_bin_dir = self.build_dir / 'bin' / 'intel64' / build_config

def copy_bin(name):
    global build_bin_dir, sysroot_bin_dir
    copytree(build_bin_dir / name, sysroot_bin_dir / name)

dll_suffix = 'd' if self.config.build_debug else ''
def copy_dll(name):
    global copy_bin, dll_suffix
    copy_bin(name + dll_suffix + '.dll')
    copy_bin(name + dll_suffix + '.pdb')

copy_bin('cache.json')
copy_dll('clDNNPlugin')
copy_dll('HeteroPlugin')
copy_dll('inference_engine')
copy_dll('inference_engine_ir_reader')
copy_dll('inference_engine_ir_v7_reader')
copy_dll('inference_engine_legacy')
copy_dll('inference_engine_transformations')  # runtime
copy_dll('inference_engine_lp_transformations')  # runtime
copy_dll('MKLDNNPlugin')  # runtime
copy_dll('myriadPlugin')  # runtime
#copy_dll('MultiDevicePlugin')  # runtime, not used
copy_dll('ngraph')
copy_bin('plugins.xml')
copy_bin('pcie-ma2x8x.elf')
copy_bin('usb-ma2x8x.mvcmd')

copytree(self.srcdir / 'inference-engine' / 'temp' / 'tbb' / 'bin', sysroot_bin_dir)
copytree(self.srcdir / 'inference-engine' / 'temp' / 'tbb', self.sysrootdir / 'tbb')

sysroot_ie_dir = prepare_dir(self.sysrootdir / 'deployment_tools' / 'inference_engine')
sysroot_ie_lib_dir = prepare_dir(sysroot_ie_dir / 'lib' / 'intel64')

copytree(self.srcdir / 'inference-engine' / 'include', sysroot_ie_dir / 'include')
if not self.config.build_debug:
    copytree(build_bin_dir / 'ngraph.lib', sysroot_ie_lib_dir / 'ngraph.lib')
    copytree(build_bin_dir / 'inference_engine.lib', sysroot_ie_lib_dir / 'inference_engine.lib')
    copytree(build_bin_dir / 'inference_engine_ir_reader.lib', sysroot_ie_lib_dir / 'inference_engine_ir_reader.lib')
    copytree(build_bin_dir / 'inference_engine_legacy.lib', sysroot_ie_lib_dir / 'inference_engine_legacy.lib')
else:
    copytree(build_bin_dir / 'ngraphd.lib', sysroot_ie_lib_dir / 'ngraphd.lib')
    copytree(build_bin_dir / 'inference_engined.lib', sysroot_ie_lib_dir / 'inference_engined.lib')
    copytree(build_bin_dir / 'inference_engine_ir_readerd.lib', sysroot_ie_lib_dir / 'inference_engine_ir_readerd.lib')
    copytree(build_bin_dir / 'inference_engine_legacyd.lib', sysroot_ie_lib_dir / 'inference_engine_legacyd.lib')

sysroot_license_dir = prepare_dir(self.sysrootdir / 'etc' / 'licenses')
copytree(self.srcdir / 'LICENSE', sysroot_license_dir / 'dldt-LICENSE')
copytree(self.sysrootdir / 'tbb/LICENSE', sysroot_license_dir / 'tbb-LICENSE')

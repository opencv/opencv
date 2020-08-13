from __future__ import division
from past.utils import old_div
sysroot_bin_dir = prepare_dir(old_div(self.sysrootdir, 'bin'))
copytree(old_div(self.build_dir, 'install'), old_div(self.sysrootdir, 'ngraph'))
#rm_one(self.sysrootdir / 'ngraph' / 'lib' / 'ngraph.dll')

build_config = 'Release' if not self.config.build_debug else 'Debug'
build_bin_dir = old_div(old_div(old_div(self.build_dir, 'bin'), 'intel64'), build_config)

def copy_bin(name):
    global build_bin_dir, sysroot_bin_dir
    copytree(old_div(build_bin_dir, name), old_div(sysroot_bin_dir, name))

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
copy_dll('inference_engine_legacy')
copy_dll('inference_engine_transformations')  # runtime
copy_dll('inference_engine_lp_transformations')  # runtime
copy_dll('MKLDNNPlugin')  # runtime
copy_dll('myriadPlugin')  # runtime
#copy_dll('MultiDevicePlugin')  # runtime, not used
copy_dll('ngraph')
copy_bin('plugins.xml')
copytree(old_div(old_div(old_div(self.build_dir, 'bin'), 'intel64'), 'pcie-ma248x.elf'), old_div(sysroot_bin_dir, 'pcie-ma248x.elf'))
copytree(old_div(old_div(old_div(self.build_dir, 'bin'), 'intel64'), 'usb-ma2x8x.mvcmd'), old_div(sysroot_bin_dir, 'usb-ma2x8x.mvcmd'))
copytree(old_div(old_div(old_div(self.build_dir, 'bin'), 'intel64'), 'usb-ma2450.mvcmd'), old_div(sysroot_bin_dir, 'usb-ma2450.mvcmd'))

copytree(old_div(old_div(old_div(old_div(self.srcdir, 'inference-engine'), 'temp'), 'tbb'), 'bin'), sysroot_bin_dir)
copytree(old_div(old_div(old_div(self.srcdir, 'inference-engine'), 'temp'), 'tbb'), old_div(self.sysrootdir, 'tbb'))

sysroot_ie_dir = prepare_dir(old_div(old_div(self.sysrootdir, 'deployment_tools'), 'inference_engine'))
sysroot_ie_lib_dir = prepare_dir(old_div(old_div(sysroot_ie_dir, 'lib'), 'intel64'))

copytree(old_div(old_div(self.srcdir, 'inference-engine'), 'include'), old_div(sysroot_ie_dir, 'include'))
if not self.config.build_debug:
    copytree(old_div(old_div(old_div(self.build_dir, 'install'), 'lib'), 'ngraph.lib'), old_div(sysroot_ie_lib_dir, 'ngraph.lib'))
    copytree(old_div(build_bin_dir, 'inference_engine.lib'), old_div(sysroot_ie_lib_dir, 'inference_engine.lib'))
    copytree(old_div(build_bin_dir, 'inference_engine_ir_reader.lib'), old_div(sysroot_ie_lib_dir, 'inference_engine_ir_reader.lib'))
    copytree(old_div(build_bin_dir, 'inference_engine_legacy.lib'), old_div(sysroot_ie_lib_dir, 'inference_engine_legacy.lib'))
else:
    copytree(old_div(old_div(old_div(self.build_dir, 'install'), 'lib'), 'ngraphd.lib'), old_div(sysroot_ie_lib_dir, 'ngraphd.lib'))
    copytree(old_div(build_bin_dir, 'inference_engined.lib'), old_div(sysroot_ie_lib_dir, 'inference_engined.lib'))
    copytree(old_div(build_bin_dir, 'inference_engine_ir_readerd.lib'), old_div(sysroot_ie_lib_dir, 'inference_engine_ir_readerd.lib'))
    copytree(old_div(build_bin_dir, 'inference_engine_legacyd.lib'), old_div(sysroot_ie_lib_dir, 'inference_engine_legacyd.lib'))

sysroot_license_dir = prepare_dir(old_div(old_div(self.sysrootdir, 'etc'), 'licenses'))
copytree(old_div(self.srcdir, 'LICENSE'), old_div(sysroot_license_dir, 'dldt-LICENSE'))
copytree(old_div(self.srcdir, 'ngraph/LICENSE'), old_div(sysroot_license_dir, 'ngraph-LICENSE'))
copytree(old_div(self.sysrootdir, 'tbb/LICENSE'), old_div(sysroot_license_dir, 'tbb-LICENSE'))

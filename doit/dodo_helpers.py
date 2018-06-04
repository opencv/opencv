#!/usr/local/bin/python3
import uuid
import os
from pathlib import Path
from configparser import ConfigParser

HERE = Path('.')
ROOT = Path('..')

class BuildInfo:
    def __init__(self):
        self._config = ConfigParser()
        config_path = HERE / 'opencv.cfg'
        self._config.read(str(config_path.resolve()))

    @property
    def script_dir(self):
        return HERE

    @property
    def root_dir(self):
        return ROOT

    @property
    def config(self):
        return self._config

    @property
    def standard_source_file_dep(self):
        file_deps = [ROOT / 'CMakeLists.txt']
        file_deps += rglobfiles(ROOT / 'include')
        file_deps += rglobfiles(ROOT / 'modules')
        return file_deps

    @property
    def output_dir(self):
        return HERE / 'builds'

    @property
    def mobile_output_dir(self):
        return self.output_dir / 'mobile'

    @property
    def desktop_output_dir(self):
        return self.output_dir / 'desktop'

    @property
    def android_output_dir(self):
        return self.mobile_output_dir / 'android'

    @property
    def ios_output_dir(self):
        return self.mobile_output_dir / 'ios'

def glob1(path,wildcard):
    """
    Perform a globbing operation and confirm we get exactly one value
    """
    paths = list(path.glob(wildcard))
    if len(paths) == 1:
        return paths[0]
    raise RuntimeError("Expected 1 path, found:" + str(paths))

def rglobfiles(path):
  """
  Enumerate all of the files recursively at a path
  """
  return [p for p in path.glob('**/*') if p.is_file()]

class CheckException(Exception):
    pass

class CheckEnvVar(object):
  """
  Check that an environment variable is set
  """
  def __init__(self, varname, helptext):
      self.varname = varname
      self.helptext = helptext

  def run(self):
      try:
          value = os.environ[self.varname]
      except KeyError:
          raise CheckException("env variable {} is not set".format(self.varname))
      return "{} is set (value = {})".format(self.varname,value)

  def help(self):
      return self.helptext

def run_checks_actions(checks):
    def run():
        for check in checks:
            try:
                message = check.run()
                print( "{}OK{}: {}".format(GREEN, RESET, message) )
            except CheckException as e:
                print( "{}FAIL{}: {}".format(RED, RESET, e) )
                for helpline in check.help().split('\n'):
                    print( "      {}".format(helpline) )
                return False
        return True
    return run

def check_env_var(varname, helptext):
    return CheckEnvVar(varname, helptext)

class MarkerFile(object):
    def __init__(self, path):
        self.path = path

    def action(self): return self.writeUuid

    def writeUuid(self):
        os.makedirs(os.path.dirname(str(self.path)), exist_ok=True)
        with open(str(self.path), 'w') as f:
            f.write(str(uuid.uuid4()))





def determine_engine_version(manifest_path):
    with open(manifest_path, "rt") as f:
        return re.search(r'android:versionName="(\d+\.\d+)"', f.read(), re.MULTILINE).group(1)


def determine_opencv_version(version_hpp_path):
    # version in 2.4 - CV_VERSION_EPOCH.CV_VERSION_MAJOR.CV_VERSION_MINOR.CV_VERSION_REVISION
    # version in master - CV_VERSION_MAJOR.CV_VERSION_MINOR.CV_VERSION_REVISION-CV_VERSION_STATUS
    with open(version_hpp_path, "rt") as f:
        data = f.read()
        major = re.search(r'^#define\W+CV_VERSION_MAJOR\W+(\d+)$', data, re.MULTILINE).group(1)
        minor = re.search(r'^#define\W+CV_VERSION_MINOR\W+(\d+)$', data, re.MULTILINE).group(1)
        revision = re.search(r'^#define\W+CV_VERSION_REVISION\W+(\d+)$', data, re.MULTILINE).group(1)
        version_status = re.search(r'^#define\W+CV_VERSION_STATUS\W+"([^"]*)"$', data, re.MULTILINE).group(1)
        return "{}.{}.{}{}".format(major, minor, revision, version_status)


def output_opencv_information():
    print(determine_opencv_version(ROOT / 'modules' / 'core' / 'include' / 'opencv2' / 'core' / 'version.hpp'))
    print(determine_engine_version(ROOT / 'platforms' / 'android' / 'service' / 'engine' / 'AndroidManifest.xml'))
    return True


def task_setup():
    build_info = BuildInfo()
    actions = [
      output_opencv_information,
      'mkdir -p {}'.format(str(build_info.output_dir.resolve()))
    ]
    return {
      'doc': 'Setup output directories...',
      'actions': actions,
      'targets': [build_info.output_dir]
    }

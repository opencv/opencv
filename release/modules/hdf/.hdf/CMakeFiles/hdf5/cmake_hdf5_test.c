#include <hdf5.h>
#include <hdf5_hl.h>
int main(void) {
  char const* info_ver = "INFO" ":" H5_VERSION;
  hid_t fid;
  fid = H5Fcreate("foo.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  return 0;
}
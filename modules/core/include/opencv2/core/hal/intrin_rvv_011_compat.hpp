// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// 0.11 -> 0.12 compatibility

#ifndef _RVV_IMPLICIT_VXRM
#define _RVV_IMPLICIT_VXRM __RISCV_VXRM_RNU
#endif

// NOTE: masked should go first to avoid extra substitution (3 arg -> 4 arg -> 5 arg)

// masked
#define __riscv_vaadd(_1, _2, _3, _4) __riscv_vaadd(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vasub(_1, _2, _3, _4) __riscv_vasub(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vaaddu(_1, _2, _3, _4) __riscv_vaaddu(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vasubu(_1, _2, _3, _4) __riscv_vasubu(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vsmul(_1, _2, _3, _4) __riscv_vsmul(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vssra(_1, _2, _3, _4) __riscv_vssra(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vssrl(_1, _2, _3, _4) __riscv_vssrl(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vnclip(_1, _2, _3, _4) __riscv_vnclip(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)
#define __riscv_vnclipu(_1, _2, _3, _4) __riscv_vnclipu(_1, _2, _3, _RVV_IMPLICIT_VXRM, _4)

// unmasked
#define __riscv_vaadd(_1, _2, _3) __riscv_vaadd(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vasub(_1, _2, _3) __riscv_vasub(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vaaddu(_1, _2, _3) __riscv_vaaddu(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vasubu(_1, _2, _3) __riscv_vasubu(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vsmul(_1, _2, _3) __riscv_vsmul(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vssra(_1, _2, _3) __riscv_vssra(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vssrl(_1, _2, _3) __riscv_vssrl(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vnclip(_1, _2, _3) __riscv_vnclip(_1, _2, _RVV_IMPLICIT_VXRM, _3)
#define __riscv_vnclipu(_1, _2, _3) __riscv_vnclipu(_1, _2, _RVV_IMPLICIT_VXRM, _3)

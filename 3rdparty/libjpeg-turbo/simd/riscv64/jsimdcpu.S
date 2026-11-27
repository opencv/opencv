/*
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2026, D. R. Commander.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* This function detects non-compliant RVV 0.7.1 hardware that reports support
 * for the V extension through HWCAP, by intentionally setting tail- and
 * mask-agnostic vector configurations that were only introduced in the RVV 0.9
 * spec.  Existing non-compliant (pre-RVV 1.0) hardware will set the VILL bit
 * in VTYPE (indicating an illegal vector configuration), which is stored in
 * the XLEN-1 bit position.  Thus a simple sign check is sufficient for
 * detection.  This workaround is inexpensive and harmless on compliant
 * hardware, but we should still consider removing it once all non-compliant
 * RVV 0.7.1 hardware is out of service.
 *
 * Reference:
 * https://riscv-optimization-guide.riseproject.dev
 */

.text
.globl  has_compliant_vsetvli

has_compliant_vsetvli:
    .option push
    .option arch, +v
    vsetvli t0, zero, e8, m1, ta, ma
    csrr    a0, vtype
    sgtz    a0, a0                # Check VILL bit

    .option pop

    ret

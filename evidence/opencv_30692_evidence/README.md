# Evidence for opencv/opencv#30692

`imgproc: vectorize float getRectSubPix with RVV for 1 and 3 channels`

This gist contains the raw measurement data behind the numbers quoted in the PR
description. Nothing here is aggregated or filtered: the small regressions and the
near-1.0 ratios are included exactly as measured.

## Tested source

`modules/imgproc/src/samplers.cpp` from this PR (SHA-256
`f13f8de0f5a86324e60cac5f43d48c05f793385f79ad152a2bee4344e4c349c3`), applied on top of
5.x `f1c1be300d10b39834fd307f23e5c2d020313a31`.

## Environment

- Board: BPI-F3 / SpacemiT X60 (RVV 1.0)
- Compiler: Ubuntu clang 17.0.2, target `riscv64-unknown-linux-gnu`
- Vector length: VLEN=256, `lanes32=8`
- The runs report real macro state at runtime, not just compile flags:
  `PASS cases=... CV_RVV=1 CV_SIMD=0 CV_SIMD_SCALABLE=1 lanes32=8`
- Benchmark: CPU affinity pinned to CPU 2, OpenCV single-threaded, OpenCL off,
  warm-up plus batch calibration to ~10 ms of work per measurement,
  timing excludes transfer, compilation and SSH.

## Correctness method

Baseline and candidate versions of the translation unit are built separately and both
are linked against the *same* prebuilt support libraries (commit
`1e21d6ed7d44c4f7ae9efd4acf302d6cbdf32bff`), so the only difference is
`samplers.cpp`. Outputs are compared through the public API.

Three configurations:

| Config | Meaning |
| --- | --- |
| `rvv-default` | default build, compiler may contract into FMA |
| `rvv-strict` | `-ffp-contract=off`, no implicit contraction |
| `scalar` | RVV path disabled, upstream scalar/universal path |

7,776 records per configuration (23,328 total), split evenly between C1 and C3,
covering float and 8u sources, ROI/step offsets, vector-width boundaries and tails,
integer and fractional centres, and rectangles fully inside / touching / outside the
image. Inputs are left byte-identical and sentinels written outside the destination
ROI are checked to be untouched.

Tolerance: 8u output must be bit-identical; float uses
`3e-4 + 3e-6 * abs(reference)`. `rvv-strict` and `scalar` are bit-exact; `rvv-default`
differs only through FMA contraction, with a max absolute error of 1.22e-4, well inside
the tolerance.

## Benchmark method

Seven benchmark rounds, alternating baseline and candidate, median of per-call times.
39 cases per round; the 8u and border cases are control groups.

## Contents

| File | What it is |
| --- | --- |
| `receipt.json` | aggregate result: records, mismatches, max error, bit-exactness per config |
| `rvv-default-comparison.json`, `rvv-strict-comparison.json`, `scalar-comparison.json` | per-configuration comparison summaries |
| `bench-0..6-baseline.csv` | raw per-case timings, baseline, one file per round |
| `bench-0..6-candidate.csv` | raw per-case timings, candidate, one file per round |
| `performance.json` | consolidated per-case medians used for the tables in the PR |
| `regression-run.log` | the new `Imgproc_GetRectSubPix.floatVectorTailsAndRoi` gtest run |
| `*-correct-candidate.log` | macro state and PASS line of the correctness run per config |
| `samplers-v3.s` | disassembly of the built translation unit, to show the RVV codegen |

CSV columns: `path,cn,width,height,border,ns_per_call,iterations`.
`path` is `8u8u` (CV_8U -> CV_8U), `8u32f` (CV_8U -> CV_32F) or `32f32f`
(CV_32F -> CV_32F); `border=1` means the source rectangle touches or crosses the
image edge.

## Scope

These results cover one board, one compiler, VLEN=256 and the listed inputs. No claim
is made for other VLENs, GCC RVV, NEON/SVE hardware, or a full-repository regression
run. The board runs link against existing support libraries rather than a freshly
built whole-library RVV OpenCV.

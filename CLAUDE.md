# OpenCV 기여 개발 환경

## 언어
- 소통: 한국어
- 코드/커밋/PR: 영어 (OpenCV 프로젝트 표준)

## 테스트 장비
- 현재: M1 iMac (ARM64, NEON 기반 — SVE 검증 불가, PR body에 명시 필요)
- 예정: M5 Mac Mini (전환 시 빌드 환경 재설정 필요)

## 디렉토리 구조
- 소스: `~/Solario/Solido/open-source/opencv/`
- 빌드(cmake 산출물): `~/Solario/Solido/open-source/.artifacts/opencv/build/`
- 기타 산출물: `~/Solario/Solido/open-source/.artifacts/opencv/{analysis,patches,notes,benchmarks}/`

## 리모트
- `origin`: https://github.com/kjg0724/opencv.git (fork)
- `upstream`: https://github.com/opencv/opencv.git
- 기여 대상 브랜치: `upstream/4.x`

---

## 개발 워크플로우

### Upstream 최신화 (PR 전 필수)
```sh
git fetch upstream
git rebase upstream/4.x          # merge 아닌 rebase — 선형 히스토리 유지
git push origin <branch> --force-with-lease
```

### 브랜치 네이밍 관례
- `fix/<module>-<설명>` 또는 `<module>/<설명>` 형태
- 예: `fix/imgproc-moments-simd`, `moments-simd-universal`

### 빌드
```sh
# 구성 (최초 1회 또는 cmake 옵션 변경 시)
cmake -S ~/Solario/Solido/open-source/opencv \
      -B ~/Solario/Solido/open-source/.artifacts/opencv/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTS=ON \
      -DBUILD_PERF_TESTS=ON \
      -DBUILD_EXAMPLES=OFF \
      -DWITH_IPP=OFF

# 빌드 (특정 모듈)
cmake --build ~/Solario/Solido/open-source/.artifacts/opencv/build \
      --target opencv_test_imgproc -j$(sysctl -n hw.logicalcpu)
```

### PR 전 로컬 체크리스트
```sh
# 1) upstream sync
git fetch upstream && git rebase upstream/4.x

# 2) 빌드 + 경고 없는지 확인
cmake --build .artifacts/opencv/build -j$(sysctl -n hw.logicalcpu) 2>&1 | tee build.log
grep -i warning build.log | grep -v "^--"

# 3) 단위 테스트
.artifacts/opencv/build/bin/opencv_test_imgproc --gtest_filter="*Moments*"

# 4) Debug 빌드 assertion 확인
cmake -B .artifacts/opencv/build-debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ...
.artifacts/opencv/build-debug/bin/opencv_test_imgproc --gtest_filter="*Moments*"
```

### CI 자동 검증 항목 (PR 올리면 자동 실행)
| 잡 | SIMD 관련성 |
|---|---|
| `macOS-ARM64` | NEON 경로 — 우리 환경과 동일 |
| `Ubuntu2404-ARM64` | NEON + SVE |
| `Linux` (x86) | SSE4/AVX2 |
| `Linux-RISC-V-Clang` | RVV |
CI는 `upstream/4.x` + PR 브랜치를 임시 머지한 상태로 테스트함.

### opencv_extra 연동
테스트 데이터 추가 시 **PR 브랜치명과 동일한 브랜치명**을 본인 opencv_extra fork에 생성해야 CI가 자동 인식.

---

## 커밋/PR 컨벤션

### 커밋 메시지
```
module: brief description of change
```
- 예: `imgproc: migrate MomentsInTile_SIMD<uchar> to universal/scalable intrinsics`
- 소문자 시작, `모듈명: 동사 + 목적어`

### PR body 구조
```markdown
## Summary
## Changes          ← 설계 근거 명시 (vpisarev용)
## Load/Compute/Store separation  ← SIMD 구조 명시
## Performance (Apple M1, NEON)   ← before/after 표 필수
## Testing
## Notes            ← SVE 미검증 등 제한사항
```

### 성능 표 형식
```
| Test | Size | Before | After | Speedup |
|------|------|--------|-------|---------|
| Moments/CV_16U/640x480 | 640x480 | X ms | Y ms | Nx |
```
측정 명령:
```sh
.artifacts/opencv/build/bin/opencv_perf_imgproc \
  --gtest_filter="*Moments*16U*" \
  --perf_min_samples=15 --perf_force_samples=15
```

---

## SIMD 코딩 컨벤션

### 가드 패턴
```cpp
// 신규 코드 — 고정폭(SSE/NEON/AVX) + 가변(SVE/RVV) 동시 지원
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // vx_load(), v_store(), v_add() 등
#endif

// 구 코드 — 마이그레이션 대상 (사용 금지)
#if CV_SIMD128
```

### 핵심 API
```cpp
vx_load(ptr)                          // 플랫폼 최적 너비 로드
vx_load_expand(ptr)                   // uchar→u16, u16→u32 확장 로드
VTraits<v_int32>::vlanes()            // 런타임 레인 수
VTraits<v_int32>::max_nlanes          // 컴파일타임 최대 (배열 할당용)
vx_cleanup()                          // 루프 후 필수 (x86 vzeroupper 등)
```

### 금지 패턴
```cpp
v_not(v_eq(a, b))          // → v_ne(a, b)
v_reinterpret_as_s32(vx_load_expand(ushort*))  // vx_load_expand(ushort*)는 v_uint32 반환
v_uint16x8, v_int32x4      // 고정폭 타입 직접 사용 금지
루프 내 div/mod             // → bit-trick으로 대체
타입만 다른 커널 N개 복붙    // → 템플릿화 필수
단일 accumulator float 누산 // → v_sum0/v_sum1 2-way accumulation
parallel_for_ in memory-bound op  // false sharing 유발
```

### Overflow 분석 (TILE_SIZE=32 기준)
| 누산 | 최대값 | 적합 타입 |
|---|---|---|
| x0: sum(src) | 32 × 65535 = 2.1M | uint32 |
| x1: sum(src×x) | 32 × 65535 × 31 = 65M | uint32 |
| x2: sum(src×x²) | 32 × 65535 × 961 ≈ 2.01G | uint32 (아슬아슬, uint64 권장) |
| x3: sum(src×x³) | 32 × 65535 × 29791 ≈ 62G | uint64 필수 |

### load-compute-store 분리 패턴 (vpisarev 필수 요구)
```cpp
// 틀림
for (...) { load(); compute(); store(); }

// 맞음 (언롤 시)
load(); load(); load(); load();
compute(); compute(); compute(); compute();
store(); store(); store(); store();
```

### SIMD 구현이 복잡하면 `something.simd.hpp`로 분리
### scalar fallback 항상 유지

---

## 테스트 작성

### 테스트 파일 위치
- 단위 테스트: `modules/<module>/test/test_<feature>.cpp`
- 성능 테스트: `modules/<module>/perf/perf_<feature>.cpp`

### 단위 테스트 패턴 (asmorkalov 요구사항)
```cpp
typedef testing::TestWithParam<tuple<Size, bool>> FeatureAccuracy;

TEST_P(FeatureAccuracy, regression)
{
    cv::RNG& rng = cv::theRNG();           // theRNG 사용
    Mat src(...);
    rng.fill(src, RNG::UNIFORM, 0, 1000);

    cv::Moments ref = /* scalar 참조 */;
    cv::Moments m   = cv::moments(src, binary);

    Mat mMat(1, 10, CV_64F, (void*)&m);
    Mat refMat(1, 10, CV_64F, (void*)&ref);
    EXPECT_LE(cv::norm(mMat, refMat, NORM_INF) /        // cv::norm(NORM_INF) 사용
              (cv::norm(refMat, NORM_INF) + 1e-10), 1e-6);
}

INSTANTIATE_TEST_CASE_P(Module_Feature, FeatureAccuracy,
    testing::Combine(
        testing::Values(Size(16,16), Size(32,32), Size(64,48), Size(320,240)),
        testing::Bool()));
```

### 테스트 실행
```sh
# 단위 테스트
.artifacts/opencv/build/bin/opencv_test_imgproc --gtest_filter="*Moments*"

# 성능 테스트
.artifacts/opencv/build/bin/opencv_perf_imgproc \
  --gtest_filter="*Moments*" \
  --perf_min_samples=15 --perf_force_samples=15
```

---

## 리뷰어 패턴

### vpisarev (Vadim Pisarevsky) — 알고리즘/설계 심층 리뷰
필수 요구사항:
- load-compute-store 분리
- 코드 중복 → 템플릿화
- 2-way accumulation (float/정확도 민감 누산)
- tail 처리 완결 (AVX2에서 tail이 길면 보조 루프 추가)
- 수치 계산 근거 명시

### asmorkalov (Alexander Smorkalov) — 코드 품질 + 최종 머지
필수 요구사항:
- 최신 universal intrinsics API 사용 (`v_ne`, `vx_load_expand_q` 등)
- `TEST_P` + `theRNG()` + `cv::norm(NORM_INF)`
- perf before/after 수치 직접 게시 (Jetson ORIN, AMD Ryzen 등에서 직접 측정)

### 리뷰 응답 원칙
- 코멘트별: 수정내용 + 수치 계산 근거 + 커밋 SHA 함께 기재
- `Resolve conversation`은 리뷰어가 재확인 후 누르도록 대기
- 무조건 동의 금지 — 이해한 근거를 설명해야 신뢰 획득
- 설계 이견 시: "근거 X입니다. 대안 Z는 어떻게 생각하시나요?" 형식

### 승인까지 일반적인 라운드 수
- 단순 SIMD 마이그레이션: 2~3 라운드, 4~7일
- 신규 최적화: 3~4 라운드, 7~15일

## Cervello

- **Forma** (내가 보는 노트): `Forma/Solido/open-source/opencv/`
- **Sostanza** (Claude 맥락·판단 재료): `Sostanza/Solido/opencv.md`
- **할일**: Obsidian Tasks (`Forma/Solario/todos/YYYY-MM-DD.md`) — Notion 사용 안 함
- 세션 시작 시 `Sostanza/Solido/opencv.md` 자동 읽기, PR 상태·벤치마크 즉시 저장

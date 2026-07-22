// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/**
 * @file aruco2detect.cl
 * @brief OpenCL kernels for GPU-accelerated ArUco marker detection.
 * @version 1.0
 * @author Francisco J. Romero-Ramirez
 * @copyright 2026 Francisco J. Romero-Ramirez
 *
 * This project is based on the g+aruco project
 * (https://github.com/kiko2r/gpu_aruco).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Kernel 1: Initialization
__kernel void preprocess_and_init(__global const uchar *img,
                                  __global const uchar *mean_img,
                                  __global uchar *thresh, __global int *labels,
                                  const int width, const int height,
                                  const uchar threshold_value) {

  // Get 2D coordinates of the execution thread
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    int idx = y * width + x;

    uchar diff_val = sub_sat(mean_img[idx], img[idx]);

    // 1. Binarization
    thresh[idx] = (diff_val > threshold_value) ? 255 : 0;
    // 2. Initialization of Union-Find (Phase 1)
    labels[idx] = idx;
  }
}

// Kernel 1: Initialization
__kernel void init_labels(__global int *labels, const int width, const int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x < width && y < height) {
    int idx = y * width + x;
    labels[idx] = idx;
  }
}

// 2. Tree Merging (Union) using atomics lock-free (Rem's Algorithm / LSL
// variant)
__constant int dx[4] = {1, 1, 0, -1};
__constant int dy[4] = {0, 1, 1, 1};
__kernel void uf_merge(__global const uchar *img, __global int *labels,
                       int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    uchar my_val = img[idx];

    // We only look forward (Right, Bottom-Right, Bottom, Bottom-Left)
    // to avoid repeating checks.
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int n_idx = ny * width + nx;
            uchar n_val = img[n_idx];

            if (my_val == n_val) {
                int u = idx;
                int v = n_idx;

                // Lock-Free Union-Find (Rem's Algorithm / LSL variant)
                // + path halving to reduce tree depth.
                while (true) {
                    // PATH HALVING: climb one level and shorten the path
                    // for threads that pass through later.
                    // atomic_min is safe here: it can only lower labels[u],
                    // never raise it, so the invariant labels[x] <= x
                    // is preserved and no cycles are created.
                    int pu = labels[u];
                    int ppu = labels[pu];
                    if (pu != ppu) atomic_min(&labels[u], ppu);
                    u = pu;

                    int pv = labels[v];
                    int ppv = labels[pv];
                    if (pv != ppv) atomic_min(&labels[v], ppv);
                    v = pv;

                    if (u == v)
                        break; // Already merged into the same group

                    // SWAP: Ensure 'u' is always the potential parent (the smaller one)
                    if (u > v) {
                        int tmp = u; u = v; v = tmp;
                    }

                    // READ-BEFORE-WRITE: Read before firing the atomic
                    int current_v = labels[v];
                    if (current_v <= u) {
                        v = current_v; // Already lowered by another thread, update and re-evaluate
                        continue;
                    }

                    // UNIFIED ATOMIC: The entire warp executes this same instruction at once
                    int old = atomic_min(&labels[v], u);

                    if (old == v) break; // Success: link established

                    // If we failed (someone modified it just before), climb the tree
                    v = old;
                }
            }
        }
    }
}

// 3. Path Compression (Flatten)
__kernel void uf_flatten(__global int *labels, __global int *total_count,
                         int width, int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= width || y >= height)
    return;
  int idx = y * width + x;
  int root = labels[idx];

  // Navigate to the final root
  while (root != labels[root]) {
    root = labels[root];
  }
  // Assign the final root to the current pixel
  labels[idx] = root;
  if (root == idx) {
    atomic_inc(total_count);
  }
}

#define MAX_EXPECTED_POINTS 16
__kernel void compact_and_sort(
    __global const int *hash_counts, __global const int *corners_in,
    __global int *valid_corners_out, volatile __global int *final_polygon_count,
    int max_points, int max_polygons_allowed, int min_area_pixels,
    __global const int *labels, int width,
    int height) { // max_points will now be configurable

  int slot = get_global_id(0);
  int num_points = hash_counts[slot];
  // We only process if we have between 4 and max_points.
  // If it's -1 or < 4, we ignore it.
  if (num_points < 4 || num_points > max_points ||
      num_points > MAX_EXPECTED_POINTS)
    return;

  int base_idx = slot * max_points * 2;

  // 1. Read the points into private registers
  int2 all_pts[MAX_EXPECTED_POINTS];
  float2 center = (float2)(0.0f, 0.0f);

  for (int i = 0; i < num_points; i++) {
    all_pts[i].x = corners_in[base_idx + i * 2];
    all_pts[i].y = corners_in[base_idx + i * 2 + 1];

    // Sum for the centroid
    center.x += (float)all_pts[i].x;
    center.y += (float)all_pts[i].y;
  }

  // 2. Calculate centroid
  center.x /= (float)num_points;
  center.y /= (float)num_points;

  // 3. Calculate the angle of each point relative to the centroid
  float angles[MAX_EXPECTED_POINTS];
  for (int i = 0; i < num_points; i++) {
    // atan2(y, x) gives the angle in radians between -PI and PI
    angles[i] =
        atan2((float)all_pts[i].y - center.y, (float)all_pts[i].x - center.x);
  }

  // 4. Sort points based on their angle (Bubble Sort in registers)
  for (int i = 0; i < num_points - 1; i++) {
    for (int j = i + 1; j < num_points; j++) {
      if (angles[i] > angles[j]) {
        // Swap angles
        float tmp_a = angles[i];
        angles[i] = angles[j];
        angles[j] = tmp_a;

        // Swap points
        int2 tmp_p = all_pts[i];
        all_pts[i] = all_pts[j];
        all_pts[j] = tmp_p;
      }
    }
  }

  // 5. Find the combination of 4 points that is convex and has the maximum area
  int best_area = -1;
  int2 best_pts[4];

  for (int i = 0; i < num_points - 3; i++) {
    for (int j = i + 1; j < num_points - 2; j++) {
      for (int k = j + 1; k < num_points - 1; k++) {
        for (int l = k + 1; l < num_points; l++) {
          int2 cand[4];
          cand[0] = all_pts[i];
          cand[1] = all_pts[j];
          cand[2] = all_pts[k];
          cand[3] = all_pts[l];

          // Check convexity and angles
          bool is_valid_shape = true;
          for (int c = 0; c < 4; c++) {
            int c_curr = c;
            int c_next = (c + 1) % 4;
            int c_next2 = (c + 2) % 4;

            int dx1 = cand[c_next].x - cand[c_curr].x;
            int dy1 = cand[c_next].y - cand[c_curr].y;
            int dx2 = cand[c_next2].x - cand[c_next].x;
            int dy2 = cand[c_next2].y - cand[c_next].y;

            // Convexity check (cross product of edges)
            int cross = dx1 * dy2 - dy1 * dx2;
            if (cross <= 0) {
              is_valid_shape = false;
              break;
            }

            // Angle check (dot product to find cosine of internal angle)
            // Vectors pointing away from the corner cand[c_next]
            float u1x = (float)(-dx1);
            float u1y = (float)(-dy1);
            float u2x = (float)(dx2);
            float u2y = (float)(dy2);

            float dot = u1x * u2x + u1y * u2y;
            float sq_norm1 = u1x * u1x + u1y * u1y;
            float sq_norm2 = u2x * u2x + u2y * u2y;

            // Prevent division by zero if points are coincident
            if (sq_norm1 < 1e-5f || sq_norm2 < 1e-5f) {
              is_valid_shape = false;
              break;
            }

            // Reject if angle is too sharp (< 18 deg) or too flat (> 162 deg)
            // This corresponds to |cos_theta| > 0.95
            // Equivalent to dot^2 > 0.95^2 * sq_norm1 * sq_norm2
            if ((dot * dot) > 0.9f * sq_norm1 * sq_norm2) {

              // Reject if angle is too sharp (< 30 deg) or too flat (> 150 deg)
              // This corresponds to |cos_theta| > 0.866
              // Equivalent to dot^2 > 0.866^2 * sq_norm1 * sq_norm2 = 0.75
              // if ((dot * dot) > 0.75f * sq_norm1 * sq_norm2) {
              is_valid_shape = false;
              break;
            }
          }

          if (is_valid_shape) {
            int double_area = (cand[0].x * cand[1].y - cand[1].x * cand[0].y) +
                              (cand[1].x * cand[2].y - cand[2].x * cand[1].y) +
                              (cand[2].x * cand[3].y - cand[3].x * cand[2].y) +
                              (cand[3].x * cand[0].y - cand[0].x * cand[3].y);
            double_area = abs(double_area);

            if (double_area > best_area) {
              best_area = double_area;
              best_pts[0] = cand[0];
              best_pts[1] = cand[1];
              best_pts[2] = cand[2];
              best_pts[3] = cand[3];
            }
          }
        }
      }
    }
  }

  // A) Area Validation
  if (best_area < (min_area_pixels * 2))
    return; // Too small or no convex quad found, discarded

  // 6. Write to output array.
  // They are already guaranteed to be in cyclic clockwise order.
  int my_out_id = atomic_add(final_polygon_count, 1);
  if (my_out_id >= max_polygons_allowed) {
    return;
  }

  int out_base = my_out_id * 8;
  for (int i = 0; i < 4; i++) {
    valid_corners_out[out_base + i * 2] = best_pts[i].x;
    valid_corners_out[out_base + i * 2 + 1] = best_pts[i].y;
  }
}

__kernel void find_corners_nms(
    __global const uchar *harris, __global const int *labels,
    __global int *corners_out,
    volatile __global int *corner_count, // No longer actively used for indexing
    volatile __global int *hash_keys, volatile __global int *hash_counts,
    int radius, // NMS window
    int HASH_SIZE, int width, int height, float harris_thresh,
    int max_points_per_hash) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // Avoid going out of image bounds when checking NMS neighbors
  if (x < radius || x >= width - radius || y < radius || y >= height - radius)
    return;
  int idx = y * width + x;

  // 1. Base Harris threshold (avoid early noise)
  float my_val = harris[idx];
  if (my_val < harris_thresh)
    return;

  int my_label = labels[idx];
  //    if (min_adj[my_label] == max_adj[my_label]) return;

  // 3. Spatial Non-Maximum Suppression (NMS)
  // Check if I am the mathematically strongest point in an 11x11 window,
  // but ONLY competing against other pixels that also touch a border.
  bool is_max = true;
  int border_label = -1;
  for (int dy = -radius; dy <= radius && is_max; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      if (dx == 0 && dy == 0)
        continue;
      int n_idx = (y + dy) * width + (x + dx);

      float neighbor_val = harris[n_idx];
      // If the neighbor is stronger (or equal and coordinate tie-breaking
      // favors it)
      if (neighbor_val > my_val ||
          (neighbor_val == my_val &&
           ((y + dy < y) || (y + dy == y && x + dx < x)))) {
        is_max = false;
        break; // A stronger pixel exists, we are no longer a local maximum
      }

      // We take the neighbor's label (P') only from our immediate neighborhood
      // to ensure we are actually touching that border.
      int n_label = labels[n_idx];
      if (n_label != my_label) {
        if (border_label == -1) {
          border_label = n_label;
        }
      }
    }
  }

  if (!is_max)
    return;

  // Form the hash (unique identifier for the pair of labels)
  uint min_l = (uint)my_label;     // min(my_label, border_label);
  uint max_l = (uint)border_label; // max(my_label, border_label);
  uint hash = min_l;
  hash ^= max_l + 0x9e3779b1 + (hash << 6) + (hash >> 2);
  // Second mixing before index:
  hash = (hash ^ (hash >> 16)) * 0x85ebca6b;
  hash = (hash ^ (hash >> 13)) * 0xc2b2ae35;
  hash = hash ^ (hash >> 16);
  if (hash == 0)
    hash = 1; // 0 means empty in the map

  // Atomic insertion into hash table (linear probing)
  // int hash_idx = (hash * 2654435761u) % HASH_SIZE;
  int hash_idx = (hash * 2654435761u) & (HASH_SIZE - 1);
  int slot = -1;
  for (int i = 0; i < 64; i++) {
    // int probe = (hash_idx + i) % HASH_SIZE;
    int probe = (hash_idx + i) & (HASH_SIZE - 1);
    int prev = atomic_cmpxchg(&hash_keys[probe], 0, hash);
    if (prev == 0 || prev == hash) {
      slot = probe;
      break;
    }
  }

  if (slot != -1) {
    // Instead of a simple add, we check the limit atomically.
    bool recorded = false;

    while (true) {
      int old_count = hash_counts[slot];
      // If the polygon is already marked as invalid (-1), abort.
      if (old_count == -1) {
        break;
      }

      // If it reached or exceeded the limit, attempt to set it to -1
      // atomically.
      if (old_count >= max_points_per_hash) {
        atomic_cmpxchg(&hash_counts[slot], old_count, -1);
        break; // Whether we succeeded in setting it to -1 or not, we abort
               // saving.
      }

      // Attempt to increment the counter
      int read_count =
          atomic_cmpxchg(&hash_counts[slot], old_count, old_count + 1);
      if (read_count == old_count) {
        // Success incrementing.
        // We have our assigned index (old_count)
        // Save the coordinate in the segment corresponding to this slot.
        int base_idx = slot * max_points_per_hash * 2;
        int my_point_idx = base_idx + (old_count * 2);

        corners_out[my_point_idx] = x;
        corners_out[my_point_idx + 1] = y;

        recorded = true;
        break;
      }
      // If there was a collision (another thread incremented simultaneously),
      // the while(true) loop repeats logic with new conditions.
    }
  }
}

inline float get_subpixel_value(__global const uchar *img, int step, int width,
                                int height, float px, float py) {
  int ix = (int)px;
  int iy = (int)py;
  if (ix < 0 || iy < 0 || ix >= width - 1 || iy >= height - 1) {
    return 0.0f;
  }
  float dx = px - ix;
  float dy = py - iy;

  int idx = iy * step + ix;
  float p00 = img[idx];
  float p01 = img[idx + 1];
  float p10 = img[idx + step];
  float p11 = img[idx + step + 1];

  float top = p00 + dx * (p01 - p00);
  float bot = p10 + dx * (p11 - p10);
  return top + dy * (bot - top);
}

inline uint hash_uint(uint x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

inline uint next_random(uint *state) {
  uint x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

inline float rand_float(uint *state) {
  return (float)next_random(state) / 4294967296.0f;
}

inline float rand_gaussian(uint *state, float mean, float stddev) {
  float u1 = rand_float(state);
  float u2 = rand_float(state);
  if (u1 < 1e-6f)
    u1 = 1e-6f;
  float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265f * u2);
  return mean + z0 * stddev;
}

inline int reflect101(int x, int S) {
  if (x < 0)
    return -x;
  if (x >= S)
    return 2 * S - 2 - x;
  return x;
}

inline void thres255Adaptive(const uchar *in_grid, uchar *out_bits, int S) {
  int off = 2;
  int thres = 5;
  for (int r = 0; r < S; r++) {
    for (int c = 0; c < S; c++) {
      int sum = 0;
      for (int ky = -off; ky <= off; ky++) {
        for (int kx = -off; kx <= off; kx++) {
          int yr = reflect101(r + ky, S);
          int xc = reflect101(c + kx, S);
          sum += (int)in_grid[yr * S + xc];
        }
      }
      int avg = (sum + 12) / 25;
      int in_val = (int)in_grid[r * S + c];
      out_bits[r * S + c] = ((avg - thres) < in_val) ? 255 : 0;
    }
  }
}

__kernel void
identify_candidates(__global const uchar *img, int step, int width, int height,
                    __global const int *valid_corners, int num_candidates,
                    __global const ulong *dict_bytes, int num_markers,
                    int marker_size, int max_correction_bits,
                    float max_erroneous_bits_in_border_rate,
                    uchar detect_color_mode, uchar gridBitSampling,
                    int max_attempts, uint seed,
                    __global const int *candidate_matched,
                    __global int *out_ids, __global int *out_rotations) {
  int thread_id = get_global_id(0);
  int total_threads = num_candidates * max_attempts;
  if (thread_id >= total_threads)
    return;

  int candidate_idx = thread_id / max_attempts;
  int attempt_idx = thread_id % max_attempts;

  // Skip if candidate was already matched in a previous dictionary
  if (candidate_matched[candidate_idx] != 0) {
    out_ids[thread_id] = -1;
    out_rotations[thread_id] = -1;
    return;
  }

  // Get the 4 corners of the candidate
  int base = candidate_idx * 8;
  float x0 = valid_corners[base + 0];
  float y0 = valid_corners[base + 1];
  float x1 = valid_corners[base + 2];
  float y1 = valid_corners[base + 3];
  float x2 = valid_corners[base + 4];
  float y2 = valid_corners[base + 5];
  float x3 = valid_corners[base + 6];
  float y3 = valid_corners[base + 7];

  int coord_key = (int)(x0 + y0 + x1 + y1 + x2 + y2 + x3 + y3);

  if (attempt_idx > 0) {
    // Seed RNG based on candidate coordinates and attempt for order invariance
    uint rnd_state =
        hash_uint(seed ^ (uint)coord_key ^ ((uint)attempt_idx * 104729));
    if (rnd_state == 0)
      rnd_state = 123456789;
    x0 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    y0 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    x1 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    y1 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    x2 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    y2 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    x3 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
    y3 += rand_gaussian(&rnd_state, 0.0f, 0.75f);
  }

  // Compute homography coefficients mapping unit square [0,1]x[0,1] to quad
  float dx1 = x1 - x2;
  float dx2 = x3 - x2;
  float sx = x0 - x1 + x2 - x3;
  float dy1 = y1 - y2;
  float dy2 = y3 - y2;
  float sy = y0 - y1 + y2 - y3;

  float det = dx1 * dy2 - dx2 * dy1;
  float a_h, b_h, c_h, d_h, e_h, f_h, g_h = 0.0f, h_h = 0.0f;

  if (fabs(det) < 1e-6f) {
    a_h = x1 - x0;
    b_h = x3 - x0;
    c_h = x0;
    d_h = y1 - y0;
    e_h = y3 - y0;
    f_h = y0;
  } else {
    g_h = (sx * dy2 - sy * dx2) / det;
    h_h = (sy * dx1 - sx * dy1) / det;
    a_h = x1 - x0 + g_h * x1;
    b_h = x3 - x0 + h_h * x3;
    c_h = x0;
    d_h = y1 - y0 + g_h * y1;
    e_h = y3 - y0 + h_h * y3;
    f_h = y0;
  }

  // Sample grid values
  uchar grid_uchar[100];
  int S = marker_size + 2;
  int N = S * S;

  for (int r = 0; r < S; r++) {
    for (int c = 0; c < S; c++) {
      if (!gridBitSampling) {
        float u = (c + 0.5f) / S;
        float v = (r + 0.5f) / S;
        float px = (a_h * u + b_h * v + c_h) / (g_h * u + h_h * v + 1.0f);
        float py = (d_h * u + e_h * v + f_h) / (g_h * u + h_h * v + 1.0f);
        float val = get_subpixel_value(img, step, width, height, px, py);
        int rounded = (int)(val + 0.5f);
        if (rounded < 0)
          rounded = 0;
        if (rounded > 255)
          rounded = 255;
        grid_uchar[r * S + c] = (uchar)rounded;
      }
      else {
        // evaluate a grid of points (rows+cols) into each bit
        float sum = 0.0f;
        float intrabitIncR = 1.0f / (float)(S * S);
        float intrabitIncC = 1.0f / (float)(S * S);

        for (int sr = 0; sr < S; sr++) {
          for (int sc = 0; sc < S; sc++) {
            // Only proceed if it is a border element (first row, last row, first col, or last col)
            if (sr == 0 || sr == S - 1 || sc == 0 || sc == S - 1) {
              float u = ((float)c / (float)S) + (0.5f + (float)sc) * intrabitIncR;
              float v = ((float)r / (float)S) + (0.5f + (float)sr) * intrabitIncC;
              float px = (a_h * u + b_h * v + c_h) / (g_h * u + h_h * v + 1.0f);
              float py = (d_h * u + e_h * v + f_h) / (g_h * u + h_h * v + 1.0f);
              sum += get_subpixel_value(img, step, width, height, px, py);
            }
          }
        }

        int rounded = (int)(sum / (float)(2 * S + 2 * S - 4) + 0.5f);
        if (rounded < 0)
          rounded = 0;
        if (rounded > 255)
          rounded = 255;
        grid_uchar[r * S + c] = (uchar)rounded;
      }
    }
  }

  int threshold = 0;
  if (attempt_idx == 3 || (attempt_idx >= 5 && attempt_idx % 2 == 1)) {
    int max_grid_val = 0;
    int max_border_val = 0;
    for (int r = 0; r < S; r++) {
      for (int c = 0; c < S; c++) {
        int val = (int)grid_uchar[r * S + c];
        if (val > max_grid_val)
          max_grid_val = val;
        if (r == 0 || r == S - 1 || c == 0 || c == S - 1) {
          if (val > max_border_val)
            max_border_val = val;
        }
      }
    }
    threshold =
        max_border_val + (int)((max_grid_val - max_border_val) * 0.20f + 0.5f);
  } else if (attempt_idx == 4 || (attempt_idx >= 5 && attempt_idx % 2 == 0)) {
    int max_grid_val = 0;
    int max_border_val = 0;
    for (int r = 0; r < S; r++) {
      for (int c = 0; c < S; c++) {
        int val = (int)grid_uchar[r * S + c];
        if (val > max_grid_val)
          max_grid_val = val;
        if (r == 0 || r == S - 1 || c == 0 || c == S - 1) {
          if (val > max_border_val)
            max_border_val = val;
        }
      }
    }
    threshold =
        max_border_val + (int)((max_grid_val - max_border_val) * 0.35f + 0.5f);
  } else {
    // Compute Otsu threshold
    int hist[256];
    for (int i = 0; i < 256; i++)
      hist[i] = 0;

    int sum_val = 0;
    for (int i = 0; i < N; i++) {
      int val = (int)grid_uchar[i];
      hist[val]++;
      sum_val += val;
    }

    float sumB = 0.0f;
    int wB = 0;
    int wF = 0;

    float varMax = 0.0f;

    for (int t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB == 0)
        continue;
      wF = N - wB;
      if (wF == 0)
        break;

      sumB += (float)(t * hist[t]);

      float mB = sumB / wB;
      float mF = ((float)sum_val - sumB) / wF;

      float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
      if (varBetween > varMax) {
        varMax = varBetween;
        threshold = t;
      }
    }
  }

  // Binarize
  uchar bits_bin[100];
  if (attempt_idx == 2) {
    thres255Adaptive(grid_uchar, bits_bin, S);
  } else {
    for (int i = 0; i < N; i++) {
      bits_bin[i] = ((int)grid_uchar[i] > threshold) ? 255 : 0;
    }
  }

  int start_mode = 0;
  int end_mode = 0;
  if (detect_color_mode == 1) {
    start_mode = 1;
    end_mode = 1;
  } else if (detect_color_mode == 2) {
    start_mode = 0;
    end_mode = 1;
  }

  int best_marker_id = -1;
  int best_rotation = -1;

  for (int mode = start_mode; mode <= end_mode; mode++) {
    uchar current_bits[100];
    for (int i = 0; i < N; i++) {
      current_bits[i] = (mode == 1) ? (255 - bits_bin[i]) : bits_bin[i];
    }

    // Validate border
    int max_border_errors =
        (int)(marker_size * marker_size * max_erroneous_bits_in_border_rate);
    int border_errors = 0;
    int border_size = 1;

    for (int y = 0; y < S; y++) {
      for (int k = 0; k < border_size; k++) {
        if (current_bits[y * S + k] != 0)
          border_errors++;
        if (current_bits[y * S + (S - 1 - k)] != 0)
          border_errors++;
      }
    }
    for (int x = border_size; x < S - border_size; x++) {
      for (int k = 0; k < border_size; k++) {
        if (current_bits[k * S + x] != 0)
          border_errors++;
        if (current_bits[(S - 1 - k) * S + x] != 0)
          border_errors++;
      }
    }

    if (border_errors > max_border_errors) {
      continue;
    }

    // Extract inner bits
    uchar inner_bits[100];
    for (int y = 0; y < marker_size; y++) {
      for (int x = 0; x < marker_size; x++) {
        inner_bits[y * marker_size + x] =
            current_bits[(y + border_size) * S + (x + border_size)] / 255;
      }
    }

    // Pack the 4 rotations of the inner bits
    ulong cand_bits[4];
    for (int rot = 0; rot < 4; rot++) {
      ulong val = 0;
      int bit_idx = 0;
      for (int r = 0; r < marker_size; r++) {
        for (int c = 0; c < marker_size; c++) {
          int src_r = r;
          int src_c = c;
          if (rot == 1) {
            src_r = marker_size - 1 - c;
            src_c = r;
          } else if (rot == 2) {
            src_r = marker_size - 1 - r;
            src_c = marker_size - 1 - c;
          } else if (rot == 3) {
            src_r = c;
            src_c = marker_size - 1 - r;
          }
          uchar bit = inner_bits[src_r * marker_size + src_c];
          val |= ((ulong)bit << bit_idx);
          bit_idx++;
        }
      }
      cand_bits[rot] = val;
    }

    // Compare with dictionary codes
    int min_distance = max_correction_bits + 1;
    for (int idx = 0; idx < num_markers; idx++) {
      ulong dbits = dict_bytes[idx];
      for (int rot = 0; rot < 4; rot++) {
        int dist = popcount(cand_bits[rot] ^ dbits);
        if (dist < min_distance) {
          min_distance = dist;
          best_marker_id = idx;
          best_rotation = rot;
        }
      }
    }

    if (min_distance <= max_correction_bits) {
      break;
    } else {
      best_marker_id = -1;
      best_rotation = -1;
    }
  }

  out_ids[thread_id] = best_marker_id;
  if (best_marker_id != -1) {
    out_rotations[thread_id] = best_rotation;
  }
}

__kernel void refine_and_collect_matches(
    __global const uchar *img, int step, int width, int height,
    __global const int *valid_corners,
    __global const int *identified_ids,
    __global const int *identified_rotations,
    __global int *candidate_matched,
    int num_candidates, int max_attempts, int dictionary_idx,
    int marker_size,
    int halfwsize, int max_iter, float eps,
    __global int *detected_count,
    __global float *detected_markers,
    int max_detected_allowed) {

    int candidate_idx = get_global_id(0);
    if (candidate_idx >= num_candidates)
        return;

    // If already matched in a previous dictionary, skip
    if (candidate_matched[candidate_idx] != 0)
        return;

    // Find if this candidate matched in the current dictionary
    int matched_attempt = -1;
    int matched_id = -1;
    int matched_rot = -1;

    for (int attempt = 0; attempt < max_attempts; attempt++) {
        int idx = candidate_idx * max_attempts + attempt;
        if (identified_ids[idx] != -1) {
            matched_attempt = attempt;
            matched_id = identified_ids[idx];
            matched_rot = identified_rotations[idx];
            break;
        }
    }

    if (matched_id == -1)
        return; // No match

    // Mark as matched so other dictionaries skip it
    candidate_matched[candidate_idx] = 1;

    // Extract the corners of the candidate
    int base = candidate_idx * 8;
    float2 corners[4];
    
    // Read corners and rotate them according to matched_rot
    for (int i = 0; i < 4; i++) {
        int src_idx = (i + 4 - matched_rot) % 4;
        corners[i].x = (float)valid_corners[base + src_idx * 2];
        corners[i].y = (float)valid_corners[base + src_idx * 2 + 1];
    }



    // Perform subpixel refinement on each of the 4 corners
    for (int c_idx = 0; c_idx < 4; c_idx++) {
        float2 init_c = corners[c_idx];
        float2 c = init_c;

        for (int iter = 0; iter < max_iter; iter++) {
            float a = 0.0f;
            float b = 0.0f;
            float cl = 0.0f;
            float d = 0.0f;
            float e = 0.0f;

            for (int dy = -halfwsize; dy <= halfwsize; dy++) {
                for (int dx = -halfwsize; dx <= halfwsize; dx++) {
                    float px_f = c.x + (float)dx;
                    float py_f = c.y + (float)dy;

                    if (px_f < 1.0f || py_f < 1.0f || px_f >= (float)width - 2.0f || py_f >= (float)height - 2.0f)
                        continue;

                    float Ix = (get_subpixel_value(img, step, width, height, px_f + 1.0f, py_f) -
                                get_subpixel_value(img, step, width, height, px_f - 1.0f, py_f)) * 0.5f;
                    float Iy = (get_subpixel_value(img, step, width, height, px_f, py_f + 1.0f) -
                                get_subpixel_value(img, step, width, height, px_f, py_f - 1.0f)) * 0.5f;

                    float wx = (float)dx / halfwsize;
                    float wy = (float)dy / halfwsize;
                    float w = exp(-wx * wx - wy * wy);

                    float IxIx = Ix * Ix * w;
                    float IxIy = Ix * Iy * w;
                    float IyIy = Iy * Iy * w;

                    float px = (float)dx;
                    float py = (float)dy;

                    a += IxIx;
                    b += IxIy;
                    cl += IyIy;

                    d += IxIx * px + IxIy * py;
                    e += IxIy * px + IyIy * py;
                }
            }

            float det = a * cl - b * b;
            if (fabs(det) < 1e-6f)
                break;

            float2 new_c;
            new_c.x = c.x + (cl * d - b * e) / det;
            new_c.y = c.y + (a * e - b * d) / det;

            float2 diff = new_c - c;
            float dist_sq = diff.x * diff.x + diff.y * diff.y;

            c = new_c;

            if (dist_sq < eps * eps)
                break;
        }

        if (fabs(c.x - init_c.x) > (float)halfwsize || fabs(c.y - init_c.y) > (float)halfwsize) {
            c = init_c;
        }

        corners[c_idx] = c;
    }

    // Allocate slot in detected_markers
    int out_idx = atomic_inc(detected_count);
    if (out_idx >= max_detected_allowed)
        return;

    int out_base = out_idx * 10;
    detected_markers[out_base + 0] = (float)matched_id;
    detected_markers[out_base + 1] = (float)dictionary_idx;
    detected_markers[out_base + 2] = corners[0].x;
    detected_markers[out_base + 3] = corners[0].y;
    detected_markers[out_base + 4] = corners[1].x;
    detected_markers[out_base + 5] = corners[1].y;
    detected_markers[out_base + 6] = corners[2].x;
    detected_markers[out_base + 7] = corners[2].y;
    detected_markers[out_base + 8] = corners[3].x;
    detected_markers[out_base + 9] = corners[3].y;
}

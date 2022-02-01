import numpy as np
import numba as nb
import cv2 as cv

from .stitching_error import StitchingError


def largest_interior_rectangle(cells):
    outline = get_outline(cells)
    adjacencies = adjacencies_all_directions(cells)
    s_map, _, saddle_candidates_map = create_maps(outline, adjacencies)
    lir1 = biggest_span_in_span_map(s_map)

    candidate_cells = cells_of_interest(saddle_candidates_map)
    s_map = span_map(adjacencies[0], adjacencies[2], candidate_cells)
    lir2 = biggest_span_in_span_map(s_map)

    lir = biggest_rectangle(lir1, lir2)
    return lir


def get_outline(cells):
    contours, hierarchy = \
        cv.findContours(cells, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # TODO support multiple contours
    # test that only one regular contour exists
    if not hierarchy.shape == (1, 1, 4) or not np.all(hierarchy == -1):
        raise StitchingError("Invalid Contour. Try without cropping.")
    contour = contours[0][:, 0, :]
    x_values = contour[:, 0].astype("uint32", order="C")
    y_values = contour[:, 1].astype("uint32", order="C")
    return x_values, y_values


@nb.njit('uint32[:,::1](uint8[:,::1], boolean)', parallel=True, cache=True)
def horizontal_adjacency(cells, direction):
    result = np.zeros(cells.shape, dtype=np.uint32)
    for y in nb.prange(cells.shape[0]):
        span = 0
        if direction:
            iterator = range(cells.shape[1]-1, -1, -1)
        else:
            iterator = range(cells.shape[1])
        for x in iterator:
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit('uint32[:,::1](uint8[:,::1], boolean)', parallel=True, cache=True)
def vertical_adjacency(cells, direction):
    result = np.zeros(cells.shape, dtype=np.uint32)
    for x in nb.prange(cells.shape[1]):
        span = 0
        if direction:
            iterator = range(cells.shape[0]-1, -1, -1)
        else:
            iterator = range(cells.shape[0])
        for y in iterator:
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit(cache=True)
def adjacencies_all_directions(cells):
    h_left2right = horizontal_adjacency(cells, 1)
    h_right2left = horizontal_adjacency(cells, 0)
    v_top2bottom = vertical_adjacency(cells, 1)
    v_bottom2top = vertical_adjacency(cells, 0)
    return h_left2right, h_right2left, v_top2bottom, v_bottom2top


@nb.njit('uint32(uint32[:])', cache=True)
def predict_vector_size(array):
    zero_indices = np.where(array == 0)[0]
    if len(zero_indices) == 0:
        if len(array) == 0:
            return 0
        return len(array)
    return zero_indices[0]


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def h_vector_top2bottom(h_adjacency, x, y):
    vector_size = predict_vector_size(h_adjacency[y:, x])
    h_vector = np.zeros(vector_size, dtype=np.uint32)
    h = np.Inf
    for p in range(vector_size):
        h = np.minimum(h_adjacency[y+p, x], h)
        h_vector[p] = h
    h_vector = np.unique(h_vector)[::-1]
    return h_vector


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def h_vector_bottom2top(h_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(h_adjacency[:y+1, x]))
    h_vector = np.zeros(vector_size, dtype=np.uint32)
    h = np.Inf
    for p in range(vector_size):
        h = np.minimum(h_adjacency[y-p, x], h)
        h_vector[p] = h
    h_vector = np.unique(h_vector)[::-1]
    return h_vector


@nb.njit(cache=True)
def h_vectors_all_directions(h_left2right, h_right2left, x, y):
    h_l2r_t2b = h_vector_top2bottom(h_left2right, x, y)
    h_r2l_t2b = h_vector_top2bottom(h_right2left, x, y)
    h_l2r_b2t = h_vector_bottom2top(h_left2right, x, y)
    h_r2l_b2t = h_vector_bottom2top(h_right2left, x, y)
    return h_l2r_t2b, h_r2l_t2b, h_l2r_b2t, h_r2l_b2t


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def v_vector_left2right(v_adjacency, x, y):
    vector_size = predict_vector_size(v_adjacency[y, x:])
    v_vector = np.zeros(vector_size, dtype=np.uint32)
    v = np.Inf
    for q in range(vector_size):
        v = np.minimum(v_adjacency[y, x+q], v)
        v_vector[q] = v
    v_vector = np.unique(v_vector)[::-1]
    return v_vector


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def v_vector_right2left(v_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(v_adjacency[y, :x+1]))
    v_vector = np.zeros(vector_size, dtype=np.uint32)
    v = np.Inf
    for q in range(vector_size):
        v = np.minimum(v_adjacency[y, x-q], v)
        v_vector[q] = v
    v_vector = np.unique(v_vector)[::-1]
    return v_vector


@nb.njit(cache=True)
def v_vectors_all_directions(v_top2bottom, v_bottom2top, x, y):
    v_l2r_t2b = v_vector_left2right(v_top2bottom, x, y)
    v_r2l_t2b = v_vector_right2left(v_top2bottom, x, y)
    v_l2r_b2t = v_vector_left2right(v_bottom2top, x, y)
    v_r2l_b2t = v_vector_right2left(v_bottom2top, x, y)
    return v_l2r_t2b, v_r2l_t2b, v_l2r_b2t, v_r2l_b2t


@nb.njit('uint32[:,:](uint32[:], uint32[:])', cache=True)
def spans(h_vector, v_vector):
    spans = np.stack((h_vector, v_vector[::-1]), axis=1)
    return spans


@nb.njit('uint32[:](uint32[:,:])', cache=True)
def biggest_span(spans):
    if len(spans) == 0:
        return np.array([0, 0], dtype=np.uint32)
    areas = spans[:, 0] * spans[:, 1]
    biggest_span_index = np.where(areas == np.amax(areas))[0][0]
    return spans[biggest_span_index]


@nb.njit(cache=True)
def spans_all_directions(h_vectors, v_vectors):
    span_l2r_t2b = spans(h_vectors[0], v_vectors[0])
    span_r2l_t2b = spans(h_vectors[1], v_vectors[1])
    span_l2r_b2t = spans(h_vectors[2], v_vectors[2])
    span_r2l_b2t = spans(h_vectors[3], v_vectors[3])
    return span_l2r_t2b, span_r2l_t2b, span_l2r_b2t, span_r2l_b2t


@nb.njit(cache=True)
def get_n_directions(spans_all_directions):
    n_directions = 1
    for spans in spans_all_directions:
        all_x_1 = np.all(spans[:, 0] == 1)
        all_y_1 = np.all(spans[:, 1] == 1)
        if not all_x_1 and not all_y_1:
            n_directions += 1
    return n_directions


@nb.njit(cache=True)
def get_xy_array(x, y, spans, mode=0):
    """0 - flip none, 1 - flip x, 2 - flip y, 3 - flip both"""
    xy = spans.copy()
    xy[:, 0] = x
    xy[:, 1] = y
    if mode == 1:
        xy[:, 0] = xy[:, 0] - spans[:, 0] + 1
    if mode == 2:
        xy[:, 1] = xy[:, 1] - spans[:, 1] + 1
    if mode == 3:
        xy[:, 0] = xy[:, 0] - spans[:, 0] + 1
        xy[:, 1] = xy[:, 1] - spans[:, 1] + 1
    return xy


@nb.njit(cache=True)
def get_xy_arrays(x, y, spans_all_directions):
    xy_l2r_t2b = get_xy_array(x, y, spans_all_directions[0], 0)
    xy_r2l_t2b = get_xy_array(x, y, spans_all_directions[1], 1)
    xy_l2r_b2t = get_xy_array(x, y, spans_all_directions[2], 2)
    xy_r2l_b2t = get_xy_array(x, y, spans_all_directions[3], 3)
    return xy_l2r_t2b, xy_r2l_t2b, xy_l2r_b2t, xy_r2l_b2t


@nb.njit(cache=True)
def point_on_outline(x, y, outline):
    x_vals, y_vals = outline
    x_true = x_vals == x
    y_true = y_vals == y
    both_true = np.logical_and(x_true, y_true)
    return np.any(both_true)


@nb.njit('Tuple((uint32[:,:,::1], uint8[:,::1], uint8[:,::1]))'
         '(UniTuple(uint32[:], 2), UniTuple(uint32[:,::1], 4))',
         parallel=True, cache=True)
def create_maps(outline, adjacencies):
    x_values, y_values = outline
    h_left2right, h_right2left, v_top2bottom, v_bottom2top = adjacencies

    shape = h_left2right.shape
    span_map = np.zeros(shape + (2,), "uint32")
    direction_map = np.zeros(shape, "uint8")
    saddle_candidates_map = np.zeros(shape, "uint8")

    for idx in nb.prange(len(x_values)):
        x, y = x_values[idx], y_values[idx]
        h_vectors = h_vectors_all_directions(h_left2right, h_right2left, x, y)
        v_vectors = v_vectors_all_directions(v_top2bottom, v_bottom2top, x, y)
        span_arrays = spans_all_directions(h_vectors, v_vectors)
        n = get_n_directions(span_arrays)
        direction_map[y, x] = n
        xy_arrays = get_xy_arrays(x, y, span_arrays)
        for direction_idx in range(4):
            xy_array = xy_arrays[direction_idx]
            span_array = span_arrays[direction_idx]
            for span_idx in range(span_array.shape[0]):
                x, y = xy_array[span_idx][0], xy_array[span_idx][1]
                w, h = span_array[span_idx][0], span_array[span_idx][1]
                if w*h > span_map[y, x, 0] * span_map[y, x, 1]:
                    span_map[y, x, :] = np.array([w, h], "uint32")
                if n == 3 and not point_on_outline(x, y, outline):
                    saddle_candidates_map[y, x] = np.uint8(255)

    return span_map, direction_map, saddle_candidates_map


def cells_of_interest(cells):
    y_vals, x_vals = cells.nonzero()
    x_vals = x_vals.astype("uint32", order="C")
    y_vals = y_vals.astype("uint32", order="C")
    return x_vals, y_vals


@nb.njit('uint32[:, :, :]'
         '(uint32[:,::1], uint32[:,::1], UniTuple(uint32[:], 2))',
         parallel=True, cache=True)
def span_map(h_adjacency_left2right,
             v_adjacency_top2bottom,
             cells_of_interest):

    x_values, y_values = cells_of_interest

    span_map = np.zeros(h_adjacency_left2right.shape + (2,), dtype=np.uint32)

    for idx in nb.prange(len(x_values)):
        x, y = x_values[idx], y_values[idx]
        h_vector = h_vector_top2bottom(h_adjacency_left2right, x, y)
        v_vector = v_vector_left2right(v_adjacency_top2bottom, x, y)
        s = spans(h_vector, v_vector)
        s = biggest_span(s)
        span_map[y, x, :] = s

    return span_map


@nb.njit('uint32[:](uint32[:, :, :])', cache=True)
def biggest_span_in_span_map(span_map):
    areas = span_map[:, :, 0] * span_map[:, :, 1]
    largest_rectangle_indices = np.where(areas == np.amax(areas))
    x = largest_rectangle_indices[1][0]
    y = largest_rectangle_indices[0][0]
    span = span_map[y, x]
    return np.array([x, y, span[0], span[1]], dtype=np.uint32)


def biggest_rectangle(*args):
    biggest_rect = np.array([0, 0, 0, 0], dtype=np.uint32)
    for rect in args:
        if rect[2] * rect[3] > biggest_rect[2] * biggest_rect[3]:
            biggest_rect = rect
    return biggest_rect

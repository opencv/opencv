#define int_tp int
#define uint_tp unsigned int

__kernel void concat(const int_tp nthreads,
                     __global const Dtype* in_data,
                     const int_tp num_concats,
                     const int_tp concat_size,
                     const int_tp top_concat_axis,
                     const int_tp bottom_concat_axis,
                     const int_tp offset_concat_axis,
                     __global Dtype* out_data) {

  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp total_concat_size = concat_size * bottom_concat_axis;
    const int_tp concat_num = index / total_concat_size;
    const int_tp concat_index = index % total_concat_size;
    const int_tp top_index = concat_index
        + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[index];
  }
}

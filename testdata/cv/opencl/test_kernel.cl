__kernel void test_kernel(__global const uchar* src, int src_step, int src_offset,
                          __global uchar* dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                          int c)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   if (x < dst_cols && y < dst_rows)
   {
       int src_idx = y * src_step + x + src_offset;
       int dst_idx = y * dst_step + x + dst_offset;
       dst[dst_idx] = src[src_idx] + c;
   }
}

#include <metal_stdlib>

using namespace metal;

constant uint padding_max_dims = 10;

struct padding_params
{
    uint count;
    uint rank;
    uint element_size;
    uint mode;
    uint input_shape[padding_max_dims];
    uint output_shape[padding_max_dims];
    uint input_strides[padding_max_dims];
    uint padding_before[padding_max_dims];
};

inline int reflect101(int coordinate, int size)
{
    if (size <= 1)
        return 0;
    const int period = 2 * (size - 1);
    int reflected = coordinate % period;
    if (reflected < 0)
        reflected += period;
    return reflected < size ? reflected : period - reflected;
}

inline uint padding_input_index(
    constant padding_params& params, uint gid, thread bool& use_fill)
{
    uint remainder = gid;
    uint input_index = 0;
    use_fill = false;
    for (int dim = int(params.rank) - 1; dim >= 0; --dim)
    {
        const int output_coordinate = int(remainder % params.output_shape[dim]);
        remainder /= params.output_shape[dim];
        int input_coordinate = output_coordinate - int(params.padding_before[dim]);
        if (input_coordinate < 0 || input_coordinate >= int(params.input_shape[dim]))
        {
            if (params.mode == 0)
            {
                use_fill = true;
                input_coordinate = 0;
            }
            else if (params.mode == 1)
            {
                input_coordinate = reflect101(input_coordinate, int(params.input_shape[dim]));
            }
            else
            {
                input_coordinate = clamp(input_coordinate, 0, int(params.input_shape[dim]) - 1);
            }
        }
        input_index += uint(input_coordinate) * params.input_strides[dim];
    }
    return input_index;
}

#define DEFINE_PADDING_KERNEL(NAME, TYPE) \
kernel void NAME( \
    device const TYPE* input [[buffer(0)]], \
    device TYPE* output [[buffer(1)]], \
    device const TYPE* fill [[buffer(2)]], \
    constant padding_params& params [[buffer(3)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    bool use_fill; \
    const uint input_index = padding_input_index(params, gid, use_fill); \
    output[gid] = use_fill ? fill[0] : input[input_index]; \
}

DEFINE_PADDING_KERNEL(kernel_padding_u8, uchar)
DEFINE_PADDING_KERNEL(kernel_padding_u16, ushort)
DEFINE_PADDING_KERNEL(kernel_padding_u32, uint)
DEFINE_PADDING_KERNEL(kernel_padding_u64, ulong)
DEFINE_PADDING_KERNEL(kernel_padding_u128, uint4)

kernel void kernel_padding_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const uchar* fill [[buffer(2)]],
    constant padding_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    bool use_fill;
    const uint input_index = padding_input_index(params, gid, use_fill);
    for (uint byte = 0; byte < params.element_size; ++byte)
        output[gid * params.element_size + byte] = use_fill ? fill[byte] :
            input[input_index * params.element_size + byte];
}

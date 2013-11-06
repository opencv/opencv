///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BORDER_REPLICATE
//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   :(i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
//BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)-1 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT101
//BORDER_REFLECT101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

__kernel void sobel3(
        __global uchar* Src,
        __global float* DstX,
        __global float* DstY,
        int width, int height,
        uint srcStride, uint dstStride,
        float scale
        )
{
    __local float lsmem[BLK_Y+2][BLK_X+2];

    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int gix = get_group_id(0);
    int giy = get_group_id(1);

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);

    lsmem[liy+1][lix+1] = convert_float(Src[ id_y * srcStride + id_x ]);

    int id_y_h = ADDR_H(id_y-1, 0,height);
    int id_y_b = ADDR_B(id_y+1, height,id_y+1);

    int id_x_l = ADDR_L(id_x-1, 0,width);
    int id_x_r = ADDR_R(id_x+1, width,id_x+1);

    if(liy==0)
    {
        lsmem[0][lix+1]=convert_float(Src[ id_y_h * srcStride + id_x ]);

        if(lix==0)
            lsmem[0][0]=convert_float(Src[ id_y_h * srcStride + id_x_l ]);
        else if(lix==BLK_X-1)
            lsmem[0][BLK_X+1]=convert_float(Src[ id_y_h * srcStride + id_x_r ]);
    }
    else if(liy==BLK_Y-1)
    {
        lsmem[BLK_Y+1][lix+1]=convert_float(Src[ id_y_b * srcStride + id_x ]);

        if(lix==0)
            lsmem[BLK_Y+1][0]=convert_float(Src[ id_y_b * srcStride + id_x_l ]);
        else if(lix==BLK_X-1)
            lsmem[BLK_Y+1][BLK_X+1]=convert_float(Src[ id_y_b * srcStride + id_x_r ]);
    }

    if(lix==0)
        lsmem[liy+1][0]    = convert_float(Src[ id_y * srcStride + id_x_l ]);
    else if(lix==BLK_X-1)
        lsmem[liy+1][BLK_X+1] = convert_float(Src[ id_y * srcStride + id_x_r ]);

    barrier(CLK_LOCAL_MEM_FENCE);

    float u1 = lsmem[liy][lix];
    float u2 = lsmem[liy][lix+1];
    float u3 = lsmem[liy][lix+2];

    float m1 = lsmem[liy+1][lix];
    float m2 = lsmem[liy+1][lix+1];
    float m3 = lsmem[liy+1][lix+2];

    float b1 = lsmem[liy+2][lix];
    float b2 = lsmem[liy+2][lix+1];
    float b3 = lsmem[liy+2][lix+2];

    //m2 * scale;//
    float dx = mad(2.0f, m3 - m1, u3 - u1 + b3 - b1 );
    DstX[ id_y * dstStride + id_x ] = dx * scale;

    float dy = mad(2.0f, b2 - u2, b1 - u1 + b3 - u3);
    DstY[ id_y * dstStride + id_x ] = dy * scale;
}
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

enum {
    Q4_0_LAYOUT_AOS = 0,
    Q4_0_LAYOUT_SOA = 1,
    Q4_0_LAYOUT_ADRENO_TRANSPOSED = 2,
    Q4_0_LAYOUT_ADRENO_MOE_TRANS4 = 3,
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nchunks = get_num_groups(0) / ne10;
    int g       = get_group_id(0);
    int i10     = g / nchunks;
    int chunk   = g - i10 * nchunks;
    int i11     = get_group_id(1);
    int i12     = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global float * dst_row = (global float *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
    global float * src_row = (global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);

    int span  = (ne00 + nchunks - 1) / nchunks;
    int start = chunk * span;
    int end   = min(start + span, ne00);

    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        global void * src0_d,
        int q4_0_layout,
        int ne01
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int block_size = sizeof(struct block_q4_0);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;
    const int nb = ne00 / QK4_0;
    const ulong row_byte_offset = (ulong)r*nb01 + (ulong)i02*nb02 + (ulong)i03*nb03;
    const ulong row_block_offset = row_byte_offset / block_size;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        const int ib = ind / 2;
        const int ih = ind & 1;
        half d;
        global float * dst_row = (global float *) ((global char *) dst + (ulong)i12*nb3 + (ulong)i11*nb2 + (ulong)i10*nb1);
        const int dst_base = ind * 16;

        if (q4_0_layout == Q4_0_LAYOUT_ADRENO_TRANSPOSED) {
            global uchar * q = (global uchar *) src0;
            global half * scales = (global half *) src0_d;
            d = scales[ib * ne01 + r];
            const float fd = (float)d;
            for (int j = 0; j < 8; ++j) {
                const int byte_idx = ih * 8 + j;
                const int col = ib * 8 + byte_idx / 2;
                const ulong q_offset = 2 * ((ulong)col * ne01 + r) + (byte_idx & 1);
                const uint packed = q[q_offset];
                dst_row[dst_base + 2*j + 0] = ((float)(packed & 0x0F) - 8.0f) * fd;
                dst_row[dst_base + 2*j + 1] = ((float)(packed >> 4)   - 8.0f) * fd;
            }
        } else if (q4_0_layout == Q4_0_LAYOUT_ADRENO_MOE_TRANS4) {
            global uchar * q = (global uchar *) src0;
            global half * scales = (global half *) src0_d;
            d = scales[(ulong)i02 * nb * ne01 + (ulong)ib * ne01 + r];
            const float fd = (float)d;
            for (int j = 0; j < 8; ++j) {
                const int byte_idx = ih * 8 + j;
                const int word_idx = byte_idx / 4;
                const int byte_in_word = byte_idx & 3;
                const ulong q_word_offset = ((ulong)i02 * nb * ne01 + (ulong)ib * ne01) * 4 + r + (ulong)word_idx * ne01;
                const uint packed = q[4 * q_word_offset + byte_in_word];
                dst_row[dst_base + 2*j + 0] = ((float)(packed & 0x0F) - 8.0f) * fd;
                dst_row[dst_base + 2*j + 1] = ((float)(packed >> 4)   - 8.0f) * fd;
            }
        } else {
            global uchar * q;
            if (q4_0_layout == Q4_0_LAYOUT_SOA) {
                global half * scales = (global half *) src0_d;
                d = scales[row_block_offset + ib];
                q = (global uchar *) src0 + (row_block_offset + ib) * (QK4_0 / 2);
            } else {
                global struct block_q4_0 * b = (global struct block_q4_0 *) ((global char *) src0 + row_byte_offset) + ib;
                d = b->d;
                q = (global uchar *) &b->qs[0];
            }

            const float fd = (float)d;
            for (int j = 0; j < 8; ++j) {
                uint packed;
                if (ih == 0) {
                    packed = (q[2*j + 0] & 0x0F) | ((q[2*j + 1] & 0x0F) << 4);
                } else {
                    packed = ((q[2*j + 0] & 0xF0) >> 4) | (q[2*j + 1] & 0xF0);
                }
                dst_row[dst_base + 2*j + 0] = ((float)(packed & 0x0F) - 8.0f) * fd;
                dst_row[dst_base + 2*j + 1] = ((float)(packed >> 4)   - 8.0f) * fd;
            }
        }
    }
}

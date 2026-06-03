// ELU(x) = x > 0 ? x : expm1(x)   (alpha = 1; matches ggml CPU op_elu)
kernel void kernel_elu_f32(
        global const float * src0,
        ulong                offset0,
        global       float * dst,
        ulong                offsetd,
        int                  n
) {
    if (get_global_id(0) >= n) {
        return;
    }
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x > 0.0f ? x : expm1(x);
}

kernel void kernel_elu_f32_4(
        global const float4 * src0,
        ulong                 offset0,
        global       float4 * dst,
        ulong                 offsetd,
        int                   n
) {
    if (get_global_id(0) >= n) {
        return;
    }
    src0 = (global float4*)((global char*)src0 + offset0);
    dst  = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = select(expm1(x), x, x > 0.0f);
}

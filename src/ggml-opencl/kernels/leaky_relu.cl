kernel void kernel_leaky_relu_f32(
        global const float * src0,
        ulong                offset0,
        global       float * dst,
        ulong                offsetd,
        int                  n,
        float                negative_slope
) {
    if (get_global_id(0) >= n) {
        return;
    }
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = fmax(x, 0.0f) + negative_slope * fmin(x, 0.0f);
}

kernel void kernel_leaky_relu_f32_4(
        global const float4 * src0,
        ulong                 offset0,
        global       float4 * dst,
        ulong                 offsetd,
        int                   n,
        float                 negative_slope
) {
    if (get_global_id(0) >= n) {
        return;
    }
    src0 = (global float4*)((global char*)src0 + offset0);
    dst  = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = fmax(x, 0.0f) + negative_slope * fmin(x, 0.0f);
}

kernel void kernel_sin_f32(
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

    dst[get_global_id(0)] = sin(src0[get_global_id(0)]);
}

kernel void kernel_sin_f32_4(
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

    dst[get_global_id(0)] = sin(src0[get_global_id(0)]);
}

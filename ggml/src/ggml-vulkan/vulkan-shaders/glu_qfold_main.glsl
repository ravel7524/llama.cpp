// glu_main.glsl with 4 dst elements per invocation and the qfold_store.glsl epilogue; every invocation runs to the end (barriers)

#define QFOLD_BINDING 3
#define QFOLD_WG 512
#include "qfold_store.glsl"

void main() {
    // the workgroup's first invocation in glu_main's flattening, plus the epilogue's (subgroup-major) index in it
    const uint qt = qfold_tid();
    const uint t = gl_WorkGroupID.z * 262144 + gl_WorkGroupID.y * 512 + gl_WorkGroupID.x * 512 + qt;
    const uint i0 = t * 4;
    const bool valid = i0 < p.N;

    vec4 v = vec4(0.0f);
    if (valid) {
        [[unroll]] for (uint j = 0; j < 4; ++j) {
            const uint i = i0 + j;

            const uint i23 = fastdiv(i, p.ne2_012mp, p.ne2_012L);
            const uint i23_offset = i23 * p.ne22*p.ne21*p.ne20;
            const uint i22 = fastdiv(i - i23_offset, p.ne2_01mp, p.ne2_01L);
            const uint i22_offset = i22*p.ne21*p.ne20;
            const uint i21 = fastdiv(i - i23_offset - i22_offset, p.ne2_0mp, p.ne2_0L);
            const uint i20 = i - i23_offset - i22_offset - i21*p.ne20;

            const uint src_idx_a = get_aoffset() + i23 * p.nb03 + i22 * p.nb02 + i21 * p.nb01 + i20 * p.nb00;
            const uint src_idx_b = get_boffset() + i23 * p.nb13 + i22 * p.nb12 + i21 * p.nb11 + i20 * p.nb10;
            const uint dst_idx = get_doffset() + i23 * p.nb23 + i22 * p.nb22 + i21 * p.nb21 + i20 * p.nb20;

            float value;
            if (p.mode == 0) {
                // Default
                const uint offset = (p.ne00 / 2) * p.nb00;
                const uint idx = src_idx_a;

                value = op(float(data_a[idx]), float(data_a[idx + offset]));
            } else if (p.mode == 1) {
                // Swapped
                const uint offset = (p.ne00 / 2) * p.nb00;
                const uint idx = src_idx_a;

                value = op(float(data_a[idx + offset]), float(data_a[idx]));
            } else {
                // Split
                value = op(float(data_a[src_idx_a]), float(data_b[src_idx_b]));
            }
            data_d[dst_idx] = D_TYPE(value);
            v[j] = float(D_TYPE(value));
        }
    }

    qfold_store(v, valid, i0 / 32, qt);
}

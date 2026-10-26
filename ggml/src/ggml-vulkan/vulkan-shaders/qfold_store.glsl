// q8_1 x4 epilogue: writes the same blocks as quantize_q8_1.comp (QBLOCK_X4), with the same arithmetic, 4 values per invocation
// without QFOLD_SUBGROUPS it has barriers: every invocation must call qfold_store (valid == false only skips the stores)
// max and the sum of rounded values are exact in any order, so both variants give the same bits
// needs types.glsl, QFOLD_BINDING and QFOLD_WG

layout (binding = QFOLD_BINDING) writeonly buffer QFOLD_Q {block_q8_1_x4 data_qfold[];};

#ifdef QFOLD_SUBGROUPS
uint qfold_tid() { return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID; }
#else
shared float qfold_sh[QFOLD_WG];
uint qfold_tid() { return gl_LocalInvocationID.x; }
#endif

// vals: the 4 values of this invocation; ib: index of their block of 32 in the flattened tensor; tid: qfold_tid()
void qfold_store(vec4 vals, const bool valid, const uint ib, const uint tid) {
    const uint iqs = tid % 8;

    // Find absolute max for each block
    const vec4 abs_vals = abs(vals);
    const float thread_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
#ifdef QFOLD_SUBGROUPS
    const float amax = subgroupClusteredMax(thread_max, 8);
#else
    qfold_sh[tid] = thread_max;
    barrier();
    [[unroll]] for (uint s = 4; s > 0; s >>= 1) {
        if (iqs < s) {
            qfold_sh[tid] = max(qfold_sh[tid], qfold_sh[tid + s]);
        }
        barrier();
    }
    const float amax = qfold_sh[tid - iqs];
    barrier();
#endif

    const float d = amax / 127.0;
    const float d_inv = d != 0.0 ? 1.0 / d : 0.0;
    vals = round(vals * d_inv);

    if (valid) {
        data_qfold[ib / 4].qs[(ib % 4) * 8 + iqs] = pack32(i8vec4(round(vals)));
    }

    // Calculate the sum for each block
    const float thread_sum = vals.x + vals.y + vals.z + vals.w;
#ifdef QFOLD_SUBGROUPS
    const float sum = subgroupClusteredAdd(thread_sum, 8);
#else
    qfold_sh[tid] = thread_sum;
    barrier();
    [[unroll]] for (uint s = 4; s > 0; s >>= 1) {
        if (iqs < s) {
            qfold_sh[tid] += qfold_sh[tid + s];
        }
        barrier();
    }
    const float sum = qfold_sh[tid];
    barrier();
#endif
    if (valid && iqs == 0) {
        data_qfold[ib / 4].ds[ib % 4] = f16vec2(vec2(d, sum * d));
    }
}

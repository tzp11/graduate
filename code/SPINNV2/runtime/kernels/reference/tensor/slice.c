#include "reference_kernels.h"

int kernel_slice(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    if (node->input_count < 3 || node->output_count != 1) return -11;
    const Spkv2TensorRecord *input_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *output_rec = ctx->tensors[node->outputs[0]].record;
    const float *input = (const float *)ctx->tensors[node->inputs[0]].data;
    float *output = (float *)ctx->tensors[node->outputs[0]].data;
    const float *starts = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *ends = (const float *)ctx->tensors[node->inputs[2]].data;
    const float *axes = node->input_count > 3 ? (const float *)ctx->tensors[node->inputs[3]].data : NULL;
    const float *steps = node->input_count > 4 ? (const float *)ctx->tensors[node->inputs[4]].data : NULL;
    const Spkv2TensorRecord *starts_rec = ctx->tensors[node->inputs[1]].record;
    size_t parameter_count = spkv2_kernel_elem_count(starts_rec);
    int start_by_axis[8] = {0};
    int step_by_axis[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < parameter_count; i++) {
        int axis = axes ? (int)axes[i] : (int)i;
        axis = spkv2_kernel_normalize_axis(axis, input_rec->rank);
        if (axis < 0 || axis >= (int)input_rec->rank) return -12;
        int start = (int)starts[i];
        int end = (int)ends[i];
        int dim = (int)input_rec->shape[axis];
        if (start < 0) start += dim;
        if (end < 0) end += dim;
        if (start < 0) start = 0;
        if (start > dim) start = dim;
        if (end < start || end > dim) return -12;
        start_by_axis[axis] = start;
        step_by_axis[axis] = steps ? (int)steps[i] : 1;
        if (step_by_axis[axis] <= 0) return -12;
    }
    size_t count = spkv2_kernel_elem_count(output_rec);
    for (size_t index = 0; index < count; index++) {
        uint32_t coordinates[8] = {0};
        spkv2_kernel_linear_to_coords(index, output_rec, coordinates);
        for (uint16_t axis = 0; axis < input_rec->rank; axis++) {
            coordinates[axis] = (uint32_t)(start_by_axis[axis] + (int)coordinates[axis] * step_by_axis[axis]);
        }
        output[index] = input[spkv2_kernel_coords_to_linear(input_rec, coordinates)];
    }
    return 0;
}

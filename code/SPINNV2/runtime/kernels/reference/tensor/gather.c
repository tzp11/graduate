#include "reference_kernels.h"

#include <math.h>
#include <string.h>

int kernel_gather(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    if (node->input_count < 2 || node->output_count != 1) return -11;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *data_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *index_rec = ctx->tensors[node->inputs[1]].record;
    const float *data = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *indices = (const float *)ctx->tensors[node->inputs[1]].data;
    float *output = (float *)ctx->tensors[node->outputs[0]].data;
    int axis = spkv2_kernel_normalize_axis(attr.axis, data_rec->rank);
    if (axis < 0 || axis >= (int)data_rec->rank) return -11;
    size_t outer = 1;
    size_t inner = 1;
    for (int i = 0; i < axis; i++) outer *= data_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < data_rec->rank; i++) inner *= data_rec->shape[i];
    size_t index_count = spkv2_kernel_elem_count(index_rec);
    size_t axis_size = data_rec->shape[axis];
    for (size_t outer_index = 0; outer_index < outer; outer_index++) {
        for (size_t index = 0; index < index_count; index++) {
            if (!isfinite(indices[index])) return -12;
            int source_index = (int)indices[index];
            if (source_index < 0) source_index += (int)axis_size;
            if (source_index < 0 || source_index >= (int)axis_size) return -12;
            memcpy(
                output + (outer_index * index_count + index) * inner,
                data + (outer_index * axis_size + (size_t)source_index) * inner,
                inner * sizeof(float));
        }
    }
    return 0;
}

#include <torch/extension.h>

torch::Tensor flash_attention_lunch(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(flash_attention_lunch), "forward");
}
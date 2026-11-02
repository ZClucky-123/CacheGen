#include "common.cuh"

std::vector<at::Tensor> calculate_cdf_and_rank(const at::Tensor&,int,bool,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> restore_cdf(const at::Tensor&,int,int,int,int,const at::Tensor&,const at::Tensor&);
void pack_mixed45(const at::Tensor&,const at::Tensor&,at::Tensor,int);
void unpack_dequant_write(const at::Tensor&,at::Tensor,int,int,int);
void dequant_write(const at::Tensor&,const at::Tensor&,at::Tensor,int,int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("preinitialize_geometry",[](){return rank_bytes_host(256,32);});
    m.def("calculate_cdf_and_rank",&calculate_cdf_and_rank);
    m.def("restore_cdf",&restore_cdf);
    m.def("pack_mixed45",&pack_mixed45);
    m.def("unpack_dequant_write",&unpack_dequant_write);
    m.def("dequant_write",&dequant_write);
}

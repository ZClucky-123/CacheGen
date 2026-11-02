#include "common.cuh"

__global__ void pack_kernel(const int8_t* sym,uint8_t* body,int layers,int n,int d,int q) {
    int p=blockIdx.y, group=blockIdx.x*256+threadIdx.x;
    if(group>=n*d/8) return;
    int width=bits_for(p,layers,q);
    uint64_t value=0;
    for(int j=0;j<8;++j) value|=uint64_t(uint8_t(sym[int64_t(p)*n*d+group*8+j]))<<(width*j);
    int64_t off=plane_offset(p,layers,q,n,d)+int64_t(group)*width;
    for(int j=0;j<width;++j) body[off+j]=uint8_t(value>>(8*j));
}

void pack_mixed45(const at::Tensor& symbols,const at::Tensor& scales,at::Tensor body,int q) {
    cuda_contiguous(symbols,at::kChar); cuda_contiguous(scales,at::kHalf); cuda_contiguous(body,at::kByte);
    TORCH_CHECK(symbols.dim()==3 && symbols.size(0)%2==0 && q>=1 && q<=3,"invalid symbol geometry/quality");
    const int layers=symbols.size(0)/2,n=symbols.size(1),d=symbols.size(2);
    TORCH_CHECK(n>=1 && n<=4096 && d%8==0 && scales.numel()==2*layers*n,"invalid pack geometry");
    TORCH_CHECK(symbols.device()==scales.device() && symbols.device()==body.device(),"device mismatch");
    int64_t scale_bytes=scales.numel()*2, expected=scale_bytes+plane_offset(2*layers,layers,q,n,d);
    TORCH_CHECK(body.numel()>=expected,"pack body too small");
    const c10::cuda::CUDAGuard guard(symbols.device());
    auto stream=c10::cuda::getCurrentCUDAStream(symbols.get_device()).stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(body.data_ptr<uint8_t>(),scales.data_ptr(),scale_bytes,cudaMemcpyDeviceToDevice,stream));
    pack_kernel<<<dim3((n*d/8+255)/256,2*layers),256,0,stream>>>(symbols.data_ptr<int8_t>(),body.data_ptr<uint8_t>()+scale_bytes,layers,n,d,q);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

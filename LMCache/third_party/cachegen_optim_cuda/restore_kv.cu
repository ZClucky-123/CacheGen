#include "common.cuh"

template<bool PACKED>
__global__ void restore_kernel(const uint8_t* input,const __half* scales,__half* output,
    int layers,int heads,int dim,int n,int total,int start,int q) {
    int p=blockIdx.y,u=blockIdx.x*256+threadIdx.x,d=heads*dim;
    if(u>=n*d) return;
    int di=u%dim,t=(u/dim)%n,h=u/(dim*n),i=t*d+h*dim+di;
    int value;
    if(PACKED) {
        int width=bits_for(p,layers,q); int64_t off=plane_offset(p,layers,q,n,d);
        int64_t byte=int64_t(i)*width/8, plane_bytes=int64_t(n)*d*width/8;
        uint16_t bits=input[off+byte];
        if(byte+1<plane_bytes) bits|=uint16_t(input[off+byte+1])<<8;
        value=(bits>>((i*width)%8))&((1<<width)-1);
    } else value=input[int64_t(p)*n*d+i];
    float gamma=float(bins_for(p,layers,q)/2-1);
    float v=__fsub_rn(float(value),gamma);
    v=__fdiv_rn(v,gamma); v=__fmul_rn(v,__half2float(scales[p*n+t]));
    int s=p/layers,l=p%layers;
    int64_t dest=((((int64_t(l)*2+s)*heads+h)*total+start+t)*dim)+di;
    output[dest]=__float2half_rn(v);
}

void validate_final(const at::Tensor& final,int n,int start,int q) {
    cuda_contiguous(final,at::kHalf);
    TORCH_CHECK(final.dim()==5 && final.size(1)==2 && n>0 && start>=0 && start+n<=final.size(3) && q>=1 && q<=3,"invalid restore geometry");
}

void unpack_dequant_write(const at::Tensor& body,at::Tensor final,int start,int n,int q) {
    cuda_contiguous(body,at::kByte); validate_final(final,n,start,q);
    TORCH_CHECK(body.device()==final.device(),"device mismatch");
    int l=final.size(0),h=final.size(2),d=final.size(4);
    TORCH_CHECK((h*d)%8==0 && body.numel()==4*l*n+plane_offset(2*l,l,q,n,h*d),"invalid packed length");
    const c10::cuda::CUDAGuard guard(final.device());
    auto stream=c10::cuda::getCurrentCUDAStream(final.get_device()).stream();
    auto scales=reinterpret_cast<const __half*>(body.data_ptr<uint8_t>());
    restore_kernel<true><<<dim3((n*h*d+255)/256,2*l),256,0,stream>>>(body.data_ptr<uint8_t>()+4*l*n,scales,reinterpret_cast<__half*>(final.data_ptr<at::Half>()),l,h,d,n,final.size(3),start,q);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequant_write(const at::Tensor& symbols,const at::Tensor& scales,at::Tensor final,int start,int q) {
    cuda_contiguous(symbols,at::kByte); cuda_contiguous(scales,at::kHalf);
    TORCH_CHECK(symbols.dim()==3,"invalid symbol rank");
    int n=symbols.size(1); validate_final(final,n,start,q);
    int l=final.size(0),h=final.size(2),d=final.size(4);
    TORCH_CHECK(symbols.size(0)==2*l && symbols.size(2)==h*d && scales.numel()==2*l*n,"invalid symbol/scale geometry");
    TORCH_CHECK(symbols.device()==scales.device() && symbols.device()==final.device(),"device mismatch");
    const c10::cuda::CUDAGuard guard(final.device());
    auto stream=c10::cuda::getCurrentCUDAStream(final.get_device()).stream();
    restore_kernel<false><<<dim3((n*h*d+255)/256,2*l),256,0,stream>>>(symbols.data_ptr<uint8_t>(),reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),reinterpret_cast<__half*>(final.data_ptr<at::Half>()),l,h,d,n,final.size(3),start,q);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include "common.cuh"

__device__ int rank_offset(int p,int layers,int q,bool full,int d,const int32_t* sizes) {
    int off=1; const int b=q==1?16:32;
    for(int i=0;i<p;++i) off+=d*sizes[full?b:bins_for(i,layers,q)-1];
    return off;
}

__global__ void cdf_rank_kernel(const int8_t* input,int16_t* cdf,uint8_t* blob,int32_t* error,
        int layers,int n,int d,int q,bool full,const uint32_t* table,const int32_t* sizes) {
    int ch=blockIdx.x*256+threadIdx.x,p=blockIdx.y;
    if(ch>=d) return;
    const int b=q==1?16:32,m=full?b:bins_for(p,layers,q)-1;
    uint16_t hist[33]={0};
    for(int t=0;t<n;++t) {
        int value=uint8_t(input[(int64_t(p)*n+t)*d+ch]);
        if(value>=b) { atomicOr(error,2); return; }
        hist[value+1]++;
    }
    for(int i=1;i<=b;++i) hist[i]+=hist[i-1];
    if(hist[m]!=n) atomicOr(error,1);
    const int bytes=sizes[m],words=(bytes+3)/4;
    U160 rank; rank.zero();
    for(int j=1;j<m;++j) rank.add(choose(table,hist[j]+j-1,j),words);
    int offset=rank_offset(p,layers,q,full,d,sizes)+ch*bytes;
    for(int j=0;j<bytes;++j) blob[offset+bytes-1-j]=uint8_t(rank.w[j/4]>>(8*(j%4)));
    if(ch==0 && p==0) blob[0]=full?0x11:0x12;
    for(int i=0;i<=b;++i) cdf[(int64_t(p)*d+ch)*(b+1)+i]=int16_t((uint32_t(65535-b)*hist[i])/n+i);
}

__global__ void restore_kernel(const uint8_t* blob,int16_t* cdf,int32_t* error,
        int layers,int n,int d,int q,bool full,const uint32_t* table,const int32_t* sizes) {
    int ch=blockIdx.x*256+threadIdx.x,p=blockIdx.y;
    if(ch>=d) return;
    int b=q==1?16:32,m=full?b:bins_for(p,layers,q)-1;
    int bytes=sizes[m],words=(bytes+3)/4,off=rank_offset(p,layers,q,full,d,sizes)+ch*bytes;
    U160 rank; rank.zero();
    for(int j=0;j<bytes;++j) rank.w[j/4]|=uint32_t(blob[off+bytes-1-j])<<(8*(j%4));
    U160 total=choose(table,n+m-1,m-1);
    if(total.le(rank,words)) { atomicOr(error,1); return; }
    int bars[33]={0},upper=n+m-2;
    for(int j=m-1;j>=2;--j) {
        if(rank.is_zero(words)) {
            for(int k=1;k<=j;++k) bars[k]=k-1;
            break;
        }
        int lo=j-1,hi=upper;
        U160 best=choose(table,lo,j);
        while(lo<hi) {
            int mid=(lo+hi+1)/2; U160 value=choose(table,mid,j);
            if(value.le(rank,words)) { lo=mid; best=value; } else hi=mid-1;
        }
        bars[j]=lo; rank.sub(best,words); upper=lo-1;
    }
    bars[1]=rank.w[0];
    for(int i=0;i<=b;++i) {
        int prefix=i==0?0:(i<m?bars[i]-i+1:n);
        cdf[(int64_t(p)*d+ch)*(b+1)+i]=int16_t((uint32_t(65535-b)*prefix)/n+i);
    }
}

void validate_tables(const at::Tensor& table,const at::Tensor& sizes,const at::Tensor& other) {
    cuda_contiguous(table,at::kInt); cuda_contiguous(sizes,at::kInt);
    TORCH_CHECK(table.numel()==288*32*5 && sizes.numel()==33,"invalid combinadic tables");
    TORCH_CHECK(table.device()==other.device() && sizes.device()==other.device(),"device mismatch");
}

std::vector<at::Tensor> calculate_cdf_and_rank(const at::Tensor& symbols,int q,bool full,const at::Tensor& table,const at::Tensor& sizes) {
    cuda_contiguous(symbols,at::kChar); validate_tables(table,sizes,symbols);
    TORCH_CHECK(symbols.dim()==3 && symbols.size(0)%2==0 && symbols.size(1)>=1 && symbols.size(1)<=256 && q>=1 && q<=3,"invalid cdf geometry");
    const c10::cuda::CUDAGuard guard(symbols.device());
    int l=symbols.size(0)/2,n=symbols.size(1),d=symbols.size(2),b=q==1?16:32;
    int bytes=1;
    for(int p=0;p<2*l;++p) bytes+=d*rank_bytes_host(n,full?b:bins_for(p,l,q)-1);
    auto cdf=at::empty({2*l,d,b+1},symbols.options().dtype(at::kShort));
    auto blob=at::empty({bytes},symbols.options().dtype(at::kByte));
    auto error=at::zeros({1},symbols.options().dtype(at::kInt));
    auto stream=c10::cuda::getCurrentCUDAStream(symbols.get_device()).stream();
    cdf_rank_kernel<<<dim3((d+255)/256,2*l),256,0,stream>>>(symbols.data_ptr<int8_t>(),cdf.data_ptr<int16_t>(),blob.data_ptr<uint8_t>(),error.data_ptr<int32_t>(),l,n,d,q,full,reinterpret_cast<const uint32_t*>(table.data_ptr<int32_t>()),sizes.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {cdf,blob,error};
}

std::vector<at::Tensor> restore_cdf(const at::Tensor& blob,int n,int q,int layers,int d,const at::Tensor& table,const at::Tensor& sizes) {
    cuda_contiguous(blob,at::kByte); validate_tables(table,sizes,blob);
    TORCH_CHECK(n>=1 && n<=256 && q>=1 && q<=3 && layers>0 && d>0 && blob.numel()>0,"invalid cdf geometry");
    const c10::cuda::CUDAGuard guard(blob.device());
    auto head=blob.narrow(0,0,1).to(at::kCPU); int tag=head.item<uint8_t>();
    TORCH_CHECK(tag==0x11 || tag==0x12,"invalid combinadic tag"); bool full=tag==0x11;
    int bytes=1,b=q==1?16:32;
    for(int p=0;p<2*layers;++p) bytes+=d*rank_bytes_host(n,full?b:bins_for(p,layers,q)-1);
    TORCH_CHECK(blob.numel()==bytes,"invalid combinadic length");
    auto cdf=at::empty({2*layers,d,b+1},blob.options().dtype(at::kShort));
    auto error=at::zeros({1},blob.options().dtype(at::kInt));
    auto stream=c10::cuda::getCurrentCUDAStream(blob.get_device()).stream();
    restore_kernel<<<dim3((d+255)/256,2*layers),256,0,stream>>>(blob.data_ptr<uint8_t>(),cdf.data_ptr<int16_t>(),error.data_ptr<int32_t>(),layers,n,d,q,full,reinterpret_cast<const uint32_t*>(table.data_ptr<int32_t>()),sizes.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {cdf,error};
}

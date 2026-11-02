#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>

// Exact geometry-only metadata, built once on the host. No device readback.
inline int rank_bytes_host(int n,int m) {
    static const auto sizes=[] {
        std::array<std::array<uint32_t,5>,288*32> table{};
        for(int x=0;x<288;++x) {
            table[x*32][0]=1;
            for(int j=1;j<=31 && j<=x;++j) {
                uint64_t carry=0;
                for(int w=0;w<5;++w) {
                    uint64_t sum=uint64_t(table[(x-1)*32+j-1][w])+table[(x-1)*32+j][w]+carry;
                    table[x*32+j][w]=uint32_t(sum); carry=sum>>32;
                }
            }
        }
        std::array<int,257*33> out{};
        for(int nn=1;nn<=256;++nn) for(int mm=2;mm<=32;++mm) {
            auto words=table[(nn+mm-1)*32+mm-1];
            uint64_t borrow=1;
            for(int w=0;w<5;++w) {
                uint64_t value=words[w]; words[w]=uint32_t(value-borrow); borrow=value<borrow;
            }
            int bits=0;
            for(int w=4;w>=0;--w) if(words[w]) {
                uint32_t value=words[w]; bits=w*32;
                while(value) {++bits; value>>=1;} break;
            }
            out[nn*33+mm]=(bits+7)/8;
        }
        return out;
    }();
    return sizes[n*33+m];
}

inline void cuda_contiguous(const at::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype,
                "expected contiguous CUDA tensor with specified dtype");
}
__host__ __device__ inline int bins_for(int p,int layers,int q) {
    int l=p%layers; bool key=p<layers;
    if(q==3) return 32;
    if(q==2) return l<(key?10:2)?32:16;
    return l<(key?20:2)?16:12;
}
__host__ __device__ inline int bits_for(int p,int layers,int q) {
    return bins_for(p,layers,q)<=16?4:5;
}
__host__ __device__ inline int64_t plane_offset(int p,int layers,int q,int n,int d) {
    int sum=0; for(int i=0;i<p;++i) sum+=bits_for(i,layers,q);
    return int64_t(n)*d/8*sum;
}
struct U160 {
    uint32_t w[5];
    __device__ void zero() { for(int i=0;i<5;++i) w[i]=0; }
    __device__ bool is_zero(int words) const {
        for(int i=0;i<words;++i) if(w[i]) return false; return true;
    }
    __device__ bool le(const U160& o,int words) const {
        for(int i=words-1;i>=0;--i) { if(w[i]!=o.w[i]) return w[i]<o.w[i]; }
        return true;
    }
    __device__ void add(const U160& o,int words) {
        uint64_t carry=0; for(int i=0;i<words;++i) { uint64_t x=uint64_t(w[i])+o.w[i]+carry; w[i]=uint32_t(x); carry=x>>32; }
    }
    __device__ void sub(const U160& o,int words) {
        uint64_t borrow=0; for(int i=0;i<words;++i) { uint64_t rhs=uint64_t(o.w[i])+borrow; uint64_t lhs=w[i]; w[i]=uint32_t(lhs-rhs); borrow=lhs<rhs; }
    }
};
__device__ inline U160 choose(const uint32_t* table,int x,int j) {
    U160 out; for(int i=0;i<5;++i) out.w[i]=table[(x*32+j)*5+i]; return out;
}

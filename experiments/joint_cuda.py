"""GPU selected-only backend. Imports are lazy so CPU validation needs no torch."""
from math import comb
import struct

from experiments.joint_codec import (EncodedFields,CGV3,schedule,rank_bytes,
    fixed_wire_size,inspect_wire,_load_payload)


class CudaCodec:
    def __init__(self,calls,device='cuda'):
        import torch
        import cachegen_optim_cuda
        self.torch,self.ext,self.calls=torch,cachegen_optim_cuda,calls
        self.device=torch.device(device)
        self._table=None; self._sizes={}; self._rank_sizes=None; self._bins={}
        self._enc_cpu={}; self._dec_cpu={}; self._dec_gpu={}
        self._enc_body=None; self._ar_buffer=None; self._ar_lengths=None
        self._symbols=None; self._pending=False; self.bytes_d2h=0; self.bytes_h2d=0; self._device_error=False

    def synchronize_safe(self,error=None):
        fatal=('illegal memory access','device-side assert','device assert','unspecified launch failure','context is destroyed')
        if self._device_error or (error is not None and any(x in str(error).lower() for x in fatal)):
            self._device_error=True
            raise RuntimeError('device_error: CUDA state is untrusted') from error
        try:
            self.torch.cuda.synchronize(self.device)
        except BaseException as sync_error:
            self._device_error=True
            raise RuntimeError('device_error: CUDA synchronization failed') from sync_error

    def preinitialize_combinadic_table(self):
        self.ext.preinitialize_geometry()
        self._combinadic(256)
        self.synchronize_safe()
        return {'device_bytes':(self._table.numel()+self._rank_sizes.numel())*4,
                'table_bytes':self._table.numel()*4,'geometry_bytes':self._rank_sizes.numel()*4}

    def workspace_snapshot(self):
        # numel/shape are host tensor metadata. No CUDA is submitted here.
        # Capture references under the GIL before numel() can yield to an
        # executor thread growing a slot. Copy metadata, never tensor bodies.
        enc_items=list(self._enc_cpu.items()); dec_items=list(self._dec_cpu.items())
        dec_gpu_items=list(self._dec_gpu.items())
        body,ar_buffer,ar_lengths,symbols=self._enc_body,self._ar_buffer,self._ar_lengths,self._symbols
        enc={f'cpu.{name}':tensor.numel() for name,tensor in enc_items}
        dec={f'cpu.{name}':tensor.numel() for name,tensor in dec_items}
        dec.update({f'gpu.{name}':tensor.numel() for name,tensor in dec_gpu_items})
        if body is not None: enc['gpu.fixed_body']=body.numel()
        if ar_buffer is not None:
            p,d,_=ar_buffer.shape; geometry=f'{p//2}:{d}'
            enc[f'gpu.ar_buffer:{geometry}']=ar_buffer.numel()
        if ar_lengths is not None:
            p,d=ar_lengths.shape; geometry=f'{p//2}:{d}'
            enc[f'gpu.ar_lengths:{geometry}']=ar_lengths.numel()*4
        if symbols is not None: dec['gpu.symbols']=symbols.numel()
        return {'encode':enc,'restore':dec}

    def reset_buffers(self):
        if self._pending: raise RuntimeError('encoding CPU fields still leased')
        self.torch.cuda.synchronize(self.device)
        self._enc_cpu.clear(); self._dec_cpu.clear(); self._dec_gpu.clear()
        self._enc_body=self._ar_buffer=self._ar_lengths=self._symbols=None

    def release_fields(self): self._pending=False

    def _check_stream(self):
        if self._device_error: raise RuntimeError('device_error: CUDA state is untrusted')
        t=self.torch
        if t.cuda.current_stream(self.device)!=t.cuda.default_stream(self.device):
            raise RuntimeError('joint codec requires default CUDA stream')

    def _combinadic(self,n):
        t=self.torch
        if self._table is None:
            # 5 little-endian uint32 limbs encoded in torch's signed int32 dtype.
            limbs=[]
            for x in range(288):
                for j in range(32):
                    value=comb(x,j) if j<=x else 0
                    for i in range(5):
                        word=(value>>(32*i))&0xffffffff
                        limbs.append(word if word<0x80000000 else word-0x100000000)
            self._table=t.tensor(limbs,dtype=t.int32,device=self.device).reshape(288,32,5)
        if self._rank_sizes is None:
            values=[[0,0]+[rank_bytes(nn,m) for m in range(2,33)] for nn in range(1,257)]
            self._rank_sizes=t.tensor(values,dtype=t.int32,device=self.device)
        if n not in self._sizes:
            self._sizes[n]=self._rank_sizes[n-1]
        return self._table,self._sizes[n]

    def _d2h(self,tensor,name,reserve=0):
        t=self.torch; flat=tensor.contiguous().view(t.uint8).flatten(); n=flat.numel()
        slot=self._enc_cpu.get(name)
        capacity=max(n,reserve)
        if slot is None or slot.numel()<capacity:
            slot=t.empty(capacity,dtype=t.uint8,pin_memory=True); self._enc_cpu[name]=slot
        out=slot[:n]; out.copy_(flat,non_blocking=False); self.bytes_d2h+=n
        return out

    def _h2d(self,blob,name,dtype=None,reserve=0):
        t=self.torch; n=len(blob)
        cpu=self._dec_cpu.get(name); gpu=self._dec_gpu.get(name)
        capacity=max(n,reserve)
        if cpu is None or cpu.numel()<capacity:
            cpu=t.empty(capacity,dtype=t.uint8,pin_memory=True); gpu=t.empty(capacity,dtype=t.uint8,device=self.device)
            self._dec_cpu[name]=cpu; self._dec_gpu[name]=gpu
        cpu[:n].copy_(t.frombuffer(bytearray(blob),dtype=t.uint8)); out=gpu[:n]
        out.copy_(cpu[:n],non_blocking=False); self.bytes_h2d+=n
        return out if dtype is None else out.view(dtype)

    def encode_gpu(self,source,q,encoding,start,total_tokens,force_full_b=False):
        self._check_stream()
        if self._pending: raise RuntimeError('encoding fields must be sealed before reusing pinned slots')
        t=self.torch
        if source.device.type!='cuda' or source.dtype!=t.float16 or source.dim()!=5 or source.shape[1]!=2:
            raise ValueError('source must be CUDA FP16 [L,2,H,N,d]')
        l,_,h,n,d=source.shape
        if not 1<=n<=256 or h*d%8 or start<0 or start+n>total_tokens: raise ValueError('invalid source geometry')
        if not bool(t.isfinite(source).all()): raise ValueError('nonfinite source')
        from lmcache.storage_backend.serde.cachegen_encoder import torch_quant_vectorized
        bins=schedule(q,l)
        key=(q,l)
        if key not in self._bins: self._bins[key]=t.tensor(bins,dtype=t.float32,device=self.device)
        # Preserve the original FP16 input and original FP32 bins promotion.
        planes=source.permute(1,0,3,2,4).reshape(2*l,n,h*d)
        symbols,scales=torch_quant_vectorized(self._bins[key],planes)
        symbols=symbols.contiguous(); scales=scales.contiguous(); self.calls['quantize']+=1
        self.bytes_d2h=0
        meta=dict(protocol='cachegen-joint-v2',q=q,encoding=encoding,shape=[l,2,h,n,d],start=start,total_tokens=total_tokens,
                  quantization_schedule=list(bins),internal_segment_max_tokens=96)
        if encoding=='fixed_bits':
            nbytes=fixed_wire_size(l,h,d,n,q)-64
            if self._enc_body is None or self._enc_body.numel()<nbytes:
                self._enc_body=t.empty(nbytes,dtype=t.uint8,device=self.device)
            body=self._enc_body[:nbytes]
            self.ext.pack_mixed45(symbols,scales,body,q); self.calls['fixed_pack']+=1
            fields=EncodedFields(meta,self._d2h(body,'fixed_body'),self)
        else:
            import torchac_cuda
            b=16 if q==1 else 32; self.calls['cdf']+=1
            if encoding=='ac_count':
                table,sizes=self._combinadic(n)
                cdf,field,flag=self.ext.calculate_cdf_and_rank(symbols,q,force_full_b,table,sizes)
                error=int(flag.item()); self.calls['rank']+=1
                if error&2: raise ValueError('symbol exceeds original CDF capacity')
                if error&1:
                    cdf,field,flag=self.ext.calculate_cdf_and_rank(symbols,q,True,table,sizes)
                    if int(flag.item()): raise ValueError('invalid full-B histogram')
                    self.calls['cdf']+=1; self.calls['rank']+=1
                    force_full_b=True
                meta['cdf_tag']=0x11 if force_full_b else 0x12
            else:
                # No combinadic table, fixed-zero check or rank work on this branch.
                cdf=torchac_cuda.calculate_cdf(symbols,b)
                field=t.cat((t.tensor([0x10],dtype=t.uint8,device=self.device),cdf.contiguous().view(t.uint8).flatten()))
                meta['cdf_tag']=0x10
            shape=(2*l,h*d,256)
            if self._ar_buffer is None or tuple(self._ar_buffer.shape)!=shape:
                self._ar_buffer=t.empty(shape,dtype=t.uint8,device=self.device)
                self._ar_lengths=t.empty((2*l,h*d),dtype=t.int32,device=self.device)
            chunks=[]; self.calls['arithmetic']+=1
            for idx,offset in enumerate(range(0,n,96)):
                nt=min(96,n-offset)
                torchac_cuda.encode_fast_new(cdf,symbols[:,offset:offset+nt,:].contiguous(),self._ar_buffer,self._ar_lengths)
                lengths=self._ar_lengths.flatten()
                holes=(256-lengths).cumsum(0).roll(1); holes[0]=0
                indexes=holes.repeat_interleave(lengths)
                indexes=indexes+t.arange(indexes.numel(),device=self.device)
                packed=self._ar_buffer.flatten()[indexes]
                chunks.append(dict(ntokens=nt,bytestream=self._d2h(packed,f'ar_stream_{idx}',2*l*h*d*256),bytestream_lengths=self._d2h(self._ar_lengths,f'ar_lengths_{idx}')))
            fields=EncodedFields(meta,dict(cdf=self._d2h(field,'cdf'),scales=self._d2h(scales,'scales'),data_chunks=chunks),self)
        t.cuda.synchronize(self.device); self._pending=True
        return fields

    def restore(self,wire,final,start):
        self._check_stream(); t=self.torch
        self.bytes_h2d=0
        meta=inspect_wire(wire); l,_,h,n,d=meta['shape']; q=meta['q']
        if meta['start']!=start or final.dtype!=t.float16 or not final.is_contiguous() or tuple(final.shape[:3])!=(l,2,h) or final.shape[-1]!=d or final.shape[3]!=meta['total_tokens']:
            raise ValueError('final KV geometry or restore prefix mismatch')
        if wire[:4]==b'M45F':
            body=self._h2d(wire[64:],'fixed_body')
            self.ext.unpack_dequant_write(body,final,start,n,q)
        else:
            import torchac_cuda
            _,ml,_=CGV3.unpack_from(wire); payload=_load_payload(wire[CGV3.size+ml:]); b=16 if q==1 else 32
            field=payload['cdf']
            if not field or field[0]!=meta['cdf_tag']: raise ValueError('CDF tag identity mismatch')
            if field[0]==0x10:
                if meta['encoding']!='ac_cdf' or len(field)!=1+2*(2*l*h*d)*(b+1): raise ValueError('raw CDF field length mismatch')
                # Validate raw boundaries before passing untrusted tables to torchac.
                values=struct.unpack('<'+'H'*((len(field)-1)//2),field[1:])
                for i in range(0,len(values),b+1):
                    row=values[i:i+b+1]
                    if row[0]!=0 or row[-1]!=65535 or any(a>=z for a,z in zip(row,row[1:])): raise ValueError('invalid raw CDF')
                cdf=self._h2d(field[1:],'cdf',t.int16).reshape(2*l,h*d,b+1)
            elif field[0] in (0x11,0x12) and meta['encoding']=='ac_count':
                table,sizes=self._combinadic(n)
                cdf,error=self.ext.restore_cdf(self._h2d(field,'cdf'),n,q,l,h*d,table,sizes)
                if int(error.item()): raise ValueError('invalid combinadic rank')
            else: raise ValueError('invalid joint CDF representation')
            if len(payload['scales'])!=4*l*n: raise ValueError('invalid scale length')
            scales=self._h2d(payload['scales'],'scales',t.float16).reshape(2*l,n,1)
            shape=(2*l,n,h*d)
            if self._symbols is None or self._symbols.numel()<2*l*n*h*d:
                self._symbols=t.empty(2*l*n*h*d,dtype=t.uint8,device=self.device)
            symbols=self._symbols[:2*l*n*h*d].reshape(shape); offset=0
            for idx,chunk in enumerate(payload['data_chunks']):
                nt=chunk['ntokens']; stream=chunk['bytestream']; lens=struct.unpack('<'+'i'*(2*l*h*d),chunk['bytestream_lengths'])
                if not 1<=nt<=96 or offset+nt>n or any(x<=0 or x>256 for x in lens) or sum(lens)!=len(stream): raise ValueError('invalid arithmetic streams')
                data=self._h2d(stream,f'ar_stream_{idx}',reserve=2*l*h*d*256)
                lengths=self._h2d(chunk['bytestream_lengths'],f'ar_lengths_{idx}',t.int32).reshape(2*l,h*d)
                prefix=lengths.flatten().cumsum(0).reshape(2*l,h*d)
                # The original decoder accepts strided output views through packed accessors.
                torchac_cuda.decode_fast_prefsum(cdf,data,prefix,symbols[:,offset:offset+nt,:])
                offset+=nt
            if offset!=n: raise ValueError('missing arithmetic tail')
            self.ext.dequant_write(symbols,scales,final,start,q)
        t.cuda.synchronize(self.device)

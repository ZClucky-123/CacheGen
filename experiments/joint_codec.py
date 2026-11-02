"""Selected-only joint codecs and exact, dependency-free CPU protocol reference.

GPU paths use the original quantizer/torchac and cachegen_optim_cuda. CPU
reference is for correctness only; it supplies no GPU cost measurements.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import comb, isfinite
import json
import pickle
import struct

M45F = struct.Struct('<4sHHIHHHHIIIIIQQQ')
CGV3 = struct.Struct('!4sIQ')
ENCODINGS = ('ac_count', 'ac_cdf', 'fixed_bits')


@lru_cache(maxsize=None)
def schedule(q, layers=32):
    if q not in (1, 2, 3): raise ValueError('invalid quality')
    if q == 3: return (32,) * (layers * 2)
    boundary, high, low = (20,16,12) if q == 1 else (10,32,16)
    return tuple(high if l < boundary else low for l in range(layers)) + tuple(high if l < 2 else low for l in range(layers))


@lru_cache(maxsize=None)
def rank_bytes(n, m):
    if not 1 <= n <= 256 or not 2 <= m <= 32: raise ValueError('invalid combinadic geometry')
    return ((comb(n+m-1,m-1)-1).bit_length()+7)//8


def rank_counts(counts):
    if not 2 <= len(counts) <= 32 or any(c < 0 for c in counts) or not 1 <= sum(counts) <= 256:
        raise ValueError('invalid counts')
    prefix = rank = 0
    for j in range(1,len(counts)):
        prefix += counts[j-1]
        rank += comb(prefix+j-1,j)
    return rank


def unrank_counts(rank,n,m):
    if not 1 <= n <= 256 or not 2 <= m <= 32 or not 0 <= rank < comb(n+m-1,m-1):
        raise ValueError('rank out of range')
    upper, u = n+m-2, rank
    bars = [0]*m
    for j in range(m-1,1,-1):
        if u==0:
            for k in range(1,j+1): bars[k]=k-1
            break
        lo,hi = j-1,upper
        while lo < hi:
            mid = (lo+hi+1)//2
            if comb(mid,j) <= u: lo = mid
            else: hi = mid-1
        bars[j] = lo
        u -= comb(lo,j)
        upper = lo-1
    bars[1]=u
    return [bars[1]] + [bars[j+1]-bars[j]-1 for j in range(1,m-1)] + [n+m-2-bars[m-1]]


def normalize_counts(counts):
    n,b = sum(counts),len(counts)
    if not n or min(counts)<0: raise ValueError('invalid counts')
    s, cdf = 0,[0]
    for i,c in enumerate(counts,1):
        s += c
        cdf.append((65535-b)*s//n+i)
    return cdf


def encode_cdf_field(counts,bins,n,b,tag=0x12):
    if len(counts) != len(bins) or any(len(c)!=b or sum(c)!=n for c in counts):
        raise ValueError('count geometry mismatch')
    if tag == 0x12 and any(any(c[bn-1:]) for c,bn in zip(counts,bins)): tag=0x11
    if tag not in (0x10,0x11,0x12): raise ValueError('invalid cdf tag')
    data = bytearray([tag])
    for c,bn in zip(counts,bins):
        if tag == 0x10: data.extend(struct.pack('<'+'H'*(b+1),*normalize_counts(c)))
        else:
            m = b if tag == 0x11 else bn-1
            data.extend(rank_counts(c[:m]).to_bytes(rank_bytes(n,m),'big'))
    return bytes(data)


def decode_cdf_field(blob,bins,n,b):
    if not blob or blob[0] not in (0x10,0x11,0x12): raise ValueError('invalid cdf tag')
    tag,offset,out = blob[0],1,[]
    expected=1+sum(2*(b+1) if tag==0x10 else rank_bytes(n,b if tag==0x11 else bn-1) for bn in bins)
    if len(blob)!=expected: raise ValueError('cdf field length mismatch')
    for bn in bins:
        size=2*(b+1) if tag==0x10 else rank_bytes(n,b if tag==0x11 else bn-1)
        if tag==0x10:
            cdf=list(struct.unpack_from('<'+'H'*(b+1),blob,offset))
            if cdf[0]!=0 or cdf[-1]!=65535 or any(a>=z for a,z in zip(cdf,cdf[1:])):
                raise ValueError('invalid raw cdf')
        else:
            m=b if tag==0x11 else bn-1
            counts=unrank_counts(int.from_bytes(blob[offset:offset+size],'big'),n,m)+[0]*(b-m)
            cdf=normalize_counts(counts)
        out.append(cdf); offset+=size
    return out


def pack_values(values,width):
    if width not in (4,5) or any(not 0<=v<(1<<width) for v in values): raise ValueError('invalid symbols')
    out=bytearray((len(values)*width+7)//8)
    for i,v in enumerate(values):
        bit=i*width; pos=bit//8; shift=bit%8
        out[pos] |= (v<<shift)&255
        if shift+width>8: out[pos+1] |= v>>(8-shift)
    return bytes(out)


def unpack_values(blob,width,count):
    if width not in (4,5) or len(blob)!=(count*width+7)//8: raise ValueError('invalid symbol length')
    return [((blob[(i*width)//8] | ((blob[(i*width)//8+1] if (i*width)//8+1<len(blob) else 0)<<8))>>((i*width)%8))&((1<<width)-1) for i in range(count)]


@lru_cache(maxsize=None)
def fixed_wire_size(layers,heads,dim,n,q):
    if heads*dim % 8: raise ValueError('channel count must be divisible by eight')
    return 64+4*layers*n+(n*heads*dim//8)*sum(4 if bn<=16 else 5 for bn in schedule(q,layers))


@lru_cache(maxsize=None)
def cdf_field_size(layers,heads,dim,n,q,tag=0x12):
    bins=schedule(q,layers); b=16 if q==1 else 32; channels=heads*dim
    if tag==0x10: return 1+2*(2*layers*channels)*(b+1)
    if tag not in (0x11,0x12): raise ValueError('invalid CDF tag')
    return 1+channels*sum(rank_bytes(n,b if tag==0x11 else bn-1) for bn in bins)


def workspace_requirements(q,encoding,N,layers,heads,dim,direction,tag=None):
    """Deterministic selected-path staging capacities, in bytes, on the CPU.

    Arithmetic stream staging reserves the original codec's 256 bytes/table
    bound for each actual <=96-token segment. Only the valid stream bytes move.
    Read-only process tables and per-job ephemeral tensors are separate costs.
    """
    if direction not in ('encode','restore') or encoding not in ENCODINGS or not 1<=N<=256:
        raise ValueError('invalid workspace direction/recipe')
    if min(layers,heads,dim)<=0 or heads*dim%8: raise ValueError('invalid workspace geometry')
    if encoding=='fixed_bits':
        body=fixed_wire_size(layers,heads,dim,N,q)-64
        return {'gpu.fixed_body':body,'cpu.fixed_body':body}
    tables=2*layers*heads*dim
    tag=(0x10 if encoding=='ac_cdf' else 0x12) if tag is None else tag
    cdf=cdf_field_size(layers,heads,dim,N,q,tag)
    if direction=='encode':
        # The original torchac output buffer is geometrical, not q-dependent.
        geometry=f'{layers}:{heads*dim}'
        need={f'gpu.ar_buffer:{geometry}':tables*256,
              f'gpu.ar_lengths:{geometry}':tables*4,'cpu.cdf':cdf,'cpu.scales':4*layers*N}
        for i in range((N+95)//96):
            need[f'cpu.ar_stream_{i}']=tables*256
            need[f'cpu.ar_lengths_{i}']=tables*4
    else:
        need={'gpu.symbols':tables*N}
        for prefix in ('cpu.','gpu.'):
            need[prefix+'cdf']=cdf-(1 if tag==0x10 else 0)
            need[prefix+'scales']=4*layers*N
            for i in range((N+95)//96):
                need[prefix+f'ar_stream_{i}']=tables*256
                need[prefix+f'ar_lengths_{i}']=tables*4
    return need


def workspace_state_from_snapshot(snapshot,requirements,direction):
    capacities=snapshot.get(direction,{})
    primary=[name for name in requirements if name in ('gpu.fixed_body','gpu.symbols') or name.startswith('gpu.ar_buffer:')]
    if any(name not in capacities for name in primary): return 'first'
    if not primary and not any(name in capacities for name in requirements): return 'first'
    if any(name not in capacities or capacities[name]<size for name,size in requirements.items()): return 'expand'
    return 'reuse'


def arithmetic_encode(symbols,cdf):
    low,high,pending=0,0xffffffff,0
    bits=[]
    def emit(bit):
        nonlocal pending
        bits.append(bit); bits.extend([1-bit]*pending); pending=0
    for symbol in symbols:
        span=high-low+1
        ch=65536 if symbol==len(cdf)-2 else cdf[symbol+1]
        high=low-1+((span*ch)>>16)
        low=low+((span*cdf[symbol])>>16)
        while True:
            if high<0x80000000: emit(0)
            elif low>=0x80000000: emit(1)
            elif low>=0x40000000 and high<0xc0000000:
                pending+=1; low-=0x40000000; high-=0x40000000
            else: break
            low=(low<<1)&0xffffffff; high=((high<<1)|1)&0xffffffff
    pending+=1; emit(0 if low<0x40000000 else 1)
    bits.extend([0]*((-len(bits))%8))
    return bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))


def arithmetic_decode(blob,cdf,n):
    bits=[(b>>i)&1 for b in blob for i in range(7,-1,-1)]; cursor=0
    def read():
        nonlocal cursor
        bit=bits[cursor] if cursor<len(bits) else 0; cursor+=1; return bit
    low,high,value=0,0xffffffff,0
    for _ in range(32): value=(value<<1)|read()
    out=[]
    for _ in range(n):
        span=high-low+1; count=((value-low+1)*65536-1)//span
        symbol=min(bisect_right(cdf,count)-1,len(cdf)-2)
        out.append(symbol)
        ch=65536 if symbol==len(cdf)-2 else cdf[symbol+1]
        high=low-1+((span*ch)>>16); low+=((span*cdf[symbol])>>16)
        while True:
            if high<0x80000000: pass
            elif low>=0x80000000: low-=0x80000000; high-=0x80000000; value-=0x80000000
            elif low>=0x40000000 and high<0xc0000000:
                low-=0x40000000; high-=0x40000000; value-=0x40000000
            else: break
            low=(low<<1)&0xffffffff; high=((high<<1)|1)&0xffffffff; value=((value<<1)|read())&0xffffffff
    return out


def _f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]
def _half(v): return struct.unpack('<e',struct.pack('<e',v))[0]


def inspect_wire(wire):
    if wire[:4]==b'M45F':
        if len(wire)<64: raise ValueError('truncated M45F header')
        magic,version,hb,flags,l,h,d,q,start,n,total,index,count,sbytes,vbytes,res=M45F.unpack_from(wire)
        if version not in (1,2) or (version==1 and q!=2) or hb!=64 or flags or res or q not in (1,2,3):
            raise ValueError('invalid M45F header')
        if min(l,h,d,n,total,count)<=0 or total>4096 or start+n>total or start<0:
            raise ValueError('invalid M45F range')
        whole=index==0 and count==1 and start==0 and n==total
        if not whole and (index*256!=start or count!=(total+255)//256 or n!=min(256,total-start)):
            raise ValueError('invalid M45F chunk range')
        if sbytes!=4*l*n or 64+sbytes+vbytes!=len(wire) or len(wire)!=fixed_wire_size(l,h,d,n,q):
            raise ValueError('invalid M45F body length')
        return dict(q=q,encoding='fixed_bits',shape=[l,2,h,n,d],start=start,total_tokens=total,
                    scale_bytes=sbytes,symbol_bytes=vbytes,version=version)
    if len(wire)<CGV3.size: raise ValueError('truncated CGV3 header')
    magic,mbytes,pbytes=CGV3.unpack_from(wire)
    if magic!=b'CGV3' or len(wire)!=CGV3.size+mbytes+pbytes: raise ValueError('invalid CGV3 frame')
    meta=json.loads(wire[CGV3.size:CGV3.size+mbytes])
    if meta.get('protocol')!='cachegen-joint-v2' or meta.get('payload_bytes')!=pbytes: raise ValueError('invalid joint metadata')
    if meta.get('q') not in (1,2,3) or meta.get('encoding') not in ('ac_count','ac_cdf'):
        raise ValueError('invalid joint quality/encoding')
    shape=meta.get('shape')
    if not isinstance(shape,list) or len(shape)!=5 or any(type(x) is not int or x<=0 for x in shape) or shape[1]!=2:
        raise ValueError('invalid joint shape')
    l,_,h,n,d=shape
    start,total=meta.get('start'),meta.get('total_tokens')
    if type(start) is not int or type(total) is not int or start<0 or not 1<=n<=256 or not 1<=total<=4096 or start+n>total or h*d%8:
        raise ValueError('invalid joint token range')
    if meta.get('quantization_schedule')!=list(schedule(meta['q'],l)):
        raise ValueError('invalid joint quantization schedule')
    valid_tags=(0x10,) if meta['encoding']=='ac_cdf' else (0x11,0x12)
    if meta.get('cdf_tag') not in valid_tags: raise ValueError('invalid joint CDF tag')
    return meta


@dataclass
class EncodedFields:
    metadata: dict
    payload: object
    owner: object = None


class JointCodec:
    def __init__(self,engine=None,device='cpu'):
        self.engine,self.device=engine,str(device)
        self.calls=Counter(quantize=0,cdf=0,rank=0,arithmetic=0,fixed_pack=0)
        self._cuda_backend=None
        self.force_full_b=False
        self.last_metrics={'bytes_d2h':0}
        self.last_restore_metrics={'bytes_h2d':0}

    def reset_buffers(self):
        if self._cuda_backend is not None: self._cuda_backend.reset_buffers()

    def synchronize_safe(self,error=None):
        """Keep source/final leases alive until trusted in-flight CUDA work ends."""
        if self._cuda_backend is not None: self._cuda_backend.synchronize_safe(error)

    def preinitialize_combinadic_table(self):
        """Explicit model/device startup; call only when ac_count is enabled."""
        if not self.device.startswith('cuda'): return {'device_bytes':0,'cpu_reference':True}
        if self._cuda_backend is None:
            from experiments.joint_cuda import CudaCodec
            self._cuda_backend=CudaCodec(self.calls,self.device)
        return self._cuda_backend.preinitialize_combinadic_table()

    def workspace_snapshot(self):
        """Read existing tensor metadata only; no allocation, kernel or transfer."""
        return self._cuda_backend.workspace_snapshot() if self._cuda_backend is not None else {'encode':{},'restore':{}}

    def workspace_state(self,q,encoding,N,layers,heads,dim,direction,tag=None):
        required=workspace_requirements(q,encoding,N,layers,heads,dim,direction,tag)
        return workspace_state_from_snapshot(self.workspace_snapshot(),required,direction)

    def encode(self,source_slice,q,encoding,start,total_tokens):
        return self.seal(self.encode_gpu(source_slice,q,encoding,start,total_tokens))

    def encode_gpu(self,source_slice,q,encoding,start,total_tokens):
        """Selected GPU work + required D2H; CPU fields only on return."""
        self.last_metrics={'bytes_d2h':0}
        if encoding not in ENCODINGS: raise ValueError('unknown encoding')
        if self.device.startswith('cuda'):
            if self._cuda_backend is None:
                from experiments.joint_cuda import CudaCodec
                self._cuda_backend=CudaCodec(self.calls,self.device)
            try:
                fields=self._cuda_backend.encode_gpu(source_slice,q,encoding,start,total_tokens,self.force_full_b)
            except BaseException as error:
                self.synchronize_safe(error)
                raise
            finally:
                self.last_metrics={'bytes_d2h':getattr(self._cuda_backend,'bytes_d2h',0)}
            return fields
        source=source_slice.tolist() if hasattr(source_slice,'tolist') else source_slice
        l,h,n,d=len(source),len(source[0][0]),len(source[0][0][0]),len(source[0][0][0][0])
        if not 1<=n<=256 or start+n>total_tokens or h*d%8: raise ValueError('invalid source geometry')
        bins=schedule(q,l); b=16 if q==1 else 32
        symbols=[]; scales=[]; self.calls['quantize']+=1
        for p,bn in enumerate(bins):
            s,layer=divmod(p,l); plane=[]; maxes=[]; gamma=bn//2-1
            for t in range(n):
                row=[_half(source[layer][s][head][t][di]) for head in range(h) for di in range(d)]
                if not all(isfinite(x) for x in row): raise ValueError('nonfinite source')
                a=max(abs(v) for v in row); factor=_f32(gamma/max(a,2**-14))
                plane.append([int(round(_f32(_f32(v*factor)+gamma))) for v in row]); maxes.append(a)
            symbols.append(plane); scales.extend(maxes)
        metadata=dict(protocol='cachegen-joint-v2',q=q,encoding=encoding,shape=[l,2,h,n,d],start=start,total_tokens=total_tokens,
                      quantization_schedule=list(bins),internal_segment_max_tokens=96)
        scale_blob=b''.join(struct.pack('<e',a) for a in scales)
        if encoding=='fixed_bits':
            self.calls['fixed_pack']+=1
            packed=b''.join(pack_values([v for row in plane for v in row],4 if bn<=16 else 5) for plane,bn in zip(symbols,bins))
            return EncodedFields(metadata,scale_blob+packed)
        self.calls['cdf']+=1
        counts=[[sum(symbols[p][t][ch]==v for t in range(n)) for v in range(b)] for p in range(2*l) for ch in range(h*d)]
        table_bins=[bn for bn in bins for ch in range(h*d)]
        if encoding=='ac_count': self.calls['rank']+=1
        cdf_field=encode_cdf_field(counts,table_bins,n,b,0x10 if encoding=='ac_cdf' else (0x11 if self.force_full_b else 0x12))
        cdfs=[normalize_counts(c) for c in counts]
        self.calls['arithmetic']+=1
        chunks=[]
        for offset in range(0,n,96):
            nt=min(96,n-offset)
            streams=[arithmetic_encode([symbols[p][t][ch] for t in range(offset,offset+nt)],cdfs[p*h*d+ch]) for p in range(2*l) for ch in range(h*d)]
            chunks.append(dict(ntokens=nt,bytestream=b''.join(streams),bytestream_lengths=struct.pack('<'+'i'*len(streams),*(len(x) for x in streams))))
        metadata['cdf_tag']=cdf_field[0]
        return EncodedFields(metadata,dict(cdf=cdf_field,scales=scale_blob,data_chunks=chunks))

    def seal(self,fields):
        """CPU-only immutable wire construction; no hidden tensor transfers."""
        try:
            return self._seal(fields)
        finally:
            if fields.owner is not None: fields.owner.release_fields()

    def _seal(self,fields):
        meta=fields.metadata
        l,_,h,n,d=meta['shape']; q=meta['q']; start=meta['start']; total=meta['total_tokens']
        if meta['encoding']=='fixed_bits':
            body=_cpu_bytes(fields.payload)
            whole=n==total and start==0
            return M45F.pack(b'M45F',2,64,0,l,h,d,q,start,n,total,0 if whole else start//256,1 if whole else (total+255)//256,4*l*n,len(body)-4*l*n,0)+body
        data=fields.payload
        data=dict(cdf=_cpu_bytes(data['cdf']),scales=_cpu_bytes(data['scales']),data_chunks=[dict(ntokens=c['ntokens'],bytestream=_cpu_bytes(c['bytestream']),bytestream_lengths=_cpu_bytes(c['bytestream_lengths'])) for c in data['data_chunks']])
        payload=pickle.dumps(data,protocol=5)
        metadata=json.dumps({**meta,'payload_bytes':len(payload)},sort_keys=True,separators=(',',':')).encode()
        return CGV3.pack(b'CGV3',len(metadata),len(payload))+metadata+payload

    def restore(self,wire,final_kv,start):
        if self.device.startswith('cuda'):
            if self._cuda_backend is None:
                from experiments.joint_cuda import CudaCodec
                self._cuda_backend=CudaCodec(self.calls,self.device)
            try:
                return self._cuda_backend.restore(wire,final_kv,start)
            except BaseException as error:
                self.synchronize_safe(error)
                raise
            finally:
                self.last_restore_metrics={'bytes_h2d':getattr(self._cuda_backend,'bytes_h2d',0)}
        self.last_restore_metrics={'bytes_h2d':0}
        meta=inspect_wire(wire); l,_,h,n,d=meta['shape']; q=meta['q']
        if meta['start']!=start: raise ValueError('restore prefix mismatch')
        bins=schedule(q,l)
        if wire[:4]==b'M45F':
            scales=struct.unpack_from('<'+'e'*(2*l*n),wire,64)
            off=64+4*l*n; symbols=[]
            for bn in bins:
                width=4 if bn<=16 else 5; size=n*h*d*width//8
                vals=unpack_values(wire[off:off+size],width,n*h*d)
                if any(v>=bn-1 for v in vals): raise ValueError('symbol outside quantizer range')
                symbols.append([vals[t*h*d:(t+1)*h*d] for t in range(n)]); off+=size
        else:
            _,ml,pl=CGV3.unpack_from(wire); payload=_load_payload(wire[CGV3.size+ml:])
            if not payload.get('cdf') or payload['cdf'][0]!=meta['cdf_tag']: raise ValueError('CDF tag identity mismatch')
            if len(payload['scales'])!=4*l*n: raise ValueError('invalid scale length')
            scales=struct.unpack('<'+'e'*(2*l*n),payload['scales'])
            b=16 if q==1 else 32
            cdfs=decode_cdf_field(payload['cdf'],[bn for bn in bins for _ in range(h*d)],n,b)
            symbols=[[[] for _ in range(n)] for p in bins]; offset=0
            for chunk in payload['data_chunks']:
                nt=chunk['ntokens']; stream=chunk['bytestream']; lens=struct.unpack('<'+'i'*(2*l*h*d),chunk['bytestream_lengths']); pos=0
                if not 1<=nt<=96 or offset+nt>n or any(x<=0 or x>256 for x in lens) or sum(lens)!=len(stream): raise ValueError('invalid arithmetic streams')
                for idx,size in enumerate(lens):
                    p,ch=divmod(idx,h*d); vals=arithmetic_decode(stream[pos:pos+size],cdfs[idx],nt); pos+=size
                    for t,v in enumerate(vals): symbols[p][offset+t].append(v)
                offset+=nt
            if offset!=n: raise ValueError('missing arithmetic tail')
        for p,bn in enumerate(bins):
            s,layer=divmod(p,l); gamma=bn//2-1
            for t in range(n):
                a=scales[p*n+t]
                for ch,v in enumerate(symbols[p][t]):
                    head,di=divmod(ch,d); value=_half(_f32(_f32(_f32(v-gamma)/gamma)*a))
                    if hasattr(final_kv,'shape'): final_kv[layer,s,head,start+t,di]=value
                    else: final_kv[layer][s][head][start+t][di]=value


def _load_payload(blob):
    # Only primitive CPU fields are accepted; global/class pickle execution is disabled.
    import io
    class CPUFieldsUnpickler(pickle.Unpickler):
        def find_class(self,module,name): raise ValueError('non-primitive joint payload')
    return CPUFieldsUnpickler(io.BytesIO(blob)).load()


def _cpu_bytes(value):
    if isinstance(value,bytes): return value
    if value.device.type!='cpu': raise ValueError('seal requires CPU fields')
    return value.contiguous().view(__import__('torch').uint8).numpy().tobytes()

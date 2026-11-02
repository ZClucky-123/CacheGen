# CacheGen 高带宽优化：混合4/5bit GPU位打包与融合恢复

版本：1.0。更新日期：2026-11-01。

本文说明混合4/5bit位打包、融合反量化与最终KV写入的原理、消息协议和GPU执行方式。

本文集中记录一种高带宽方案：保留 CacheGen2 的原量化符号和最大值参数，用混合4/5bit GPU位打包替换算术编码；接收端将解包、FP32反量化、FP16转换和最终KV写入融合，减少编解码计算、中间张量和重复复制。

阅读顺序是“原流程及其数据 → 字节构成与动机 → 位打包和融合恢复 → 固定协议与执行方式 → 完整字节与成本边界”。原始时间列为历史实测，协议字节列为确定值，成本上限由耗时关系推导。

## 1 基线与固定范围

| 项目 | 固定配置 |
| --- | --- |
| 仓库与分支 | ZClucky-123/CacheGen，mini-repro-tinyllama |
| 源码参考提交 | d6b6626014c30b64011e4348c02510fdfbdb7204 |
| 历史批次 | results/mistral/larger/v3-larger-stable-20261025 |
| 模型 | mistral-community/Mistral-7B-v0.2，revision 80cf49b18de6354fd8a6d1e81c177dd830e05eea |
| 权重与KV | FastChat8权重，FP16 KV，单序列 |
| GPU | RTX4090D 24GB，计算能力8.9 |
| 软件环境 | Python 3.10.12，PyTorch 2.3.1+cu121，CUDA Toolkit 12.1，Transformers 4.42.3 |
| 量化 | cachegen2原torch_quant_vectorized及原最大值参数 |
| 实现栈 | C++/CUDA扩展，以torch.utils.cpp_extension.CUDAExtension构建 |
| 新方法名 | mixed45 |
| 新消息版本 | M45F v1，64字节定长头，定义见第6节 |
| 默认传输方式 | 256-token应用块，chunk_pipeline，queue_capacity=1 |
| 其他消息组织 | 相同256-token块的chunk_serial；单个4096-token消息的whole |
| 输入几何 | 32层、8个KV heads、head维度128；完整长度1～4096 tokens |
| 历史实测长度 | 1024、2048、4096 tokens |
| 网络 | 原CPU socket传输，历史参照条件为10Gbps目标带宽，实际吞吐由goodput表示 |
| GPU调度 | 主线程、PyTorch默认CUDA stream；每块恢复完成后返回回调 |

实现范围固定为直接量化K/V的旧codec。新paper_compression、anchor/delta处理、按head选择编码方式和在线自适应选择器不进入v1。论文方法另见 [论文方法对照](../../docs/reproduction/PAPER_ALIGNMENT.md)。

本方法保持原量化符号、FP16最大值和最终恢复KV逐位一致，通过减少编解码计算、中间张量和重复复制，降低原始KV就绪到首token的处理成本。

## 2 统一符号与应用块规格

### 2.1 符号表

| 符号 | 含义 | 本文主例 |
| --- | --- | ---: |
| $L$ | 模型层数 | 32 |
| $H_{\rm KV}$ | 每层KV head数 | 8 |
| $d_{\rm head}$ | 每个head的维度 | 128 |
| $D$ | 一层K或V的通道数，$H_{\rm KV}d_{\rm head}$ | 1024 |
| $T$ | 完整上下文token数 | 4096 |
| $N$ | 当前应用块token数 | 256；整包时为4096 |
| $A$ | 应用块数 | 16；整包时为1 |
| $p$ | K/V层平面编号，K全部层在前、V全部层在后 | $0,\ldots,63$ |
| $B_p$ | 原量化函数的bins配置 | 16或32 |
| $\gamma_p$ | 量化偏移和恢复系数，$\lfloor B_p/2\rfloor-1$ | 7或15 |
| $Q_p$ | 实际可表示的整数档位数，$2\gamma_p+1$ | 15或31 |
| $b_p$ | 该层符号的固定位宽，$\lceil\log_2Q_p\rceil$ | 4或5bit |
| $a_{p,t}$ | 某K/V层、某token的最大绝对值参数 | 原FP16最大值 |
| $S_{\rm ar}$、$S_{\rm fix}$ | 算术编码和定长打包的完整消息字节 | 依据具体块长计量 |
| $\beta$ | 有效网络吞吐，单位bit/s | 由传输实测估计 |

bins=16对应15个实际档位，bins=32对应31个实际档位，分别仍需4bit和5bit。量化配置、有效档位数和位宽分别定义。

### 2.2 一个应用块包含哪些数据

~~~text
4096-token上下文的KV
  ├─ 块 0：token位置    0～255
  ├─ 块 1：token位置  256～511
  │       …
  └─ 块15：token位置 3840～4095

每块输入：[层, K/V, KV heads, tokens, head维度]
          [32,  2,     8,       256,      128]

按K全部层、V全部层整理后：[2L, N, D] = [64, 256, 1024]
最大值参数合计：         [2L, N, 1] = [64, 256,    1]
~~~

这些KV由完整上下文prefill后切片得到。每个token的最大值参数按该层全部1024个通道计算；4/5bit选择按K/V层平面确定。

原算术路径在256-token应用块内采用96、96、64三个安全段。定长表示的偏移由形状和位宽直接计算，主运行方式沿用256-token应用块边界；整包控制组只有一个应用块。

## 3 原方法：量化、CDF、算术编码与恢复

### 3.1 量化：符号与最大值参数

对固定的K/V层平面和token，取通道向量 $x_d$：

$$
a=\max_d|x_d|,\qquad
\gamma=\lfloor {\rm bins}/2\rfloor-1,
$$

$$
q_d=\operatorname{round}
\left(x_d\frac{\gamma}{\max(a,\epsilon)}+\gamma\right).
$$

实现用输入dtype的最小正规数保护零分母，输出整数符号和原最大值参数。正常符号范围为 $0,\ldots,2\gamma$。

接收端保持原FP32运算顺序：

$$
\hat{x}_d=\operatorname{FP16}
\left(\frac{\operatorname{FP32}(q_d)-\gamma}{\gamma}a\right).
$$

例子：bins=32、$a=1.5$、$x=0.5$，则 $\gamma=15$、$q=20$，恢复为 $(20-15)/15\times1.5=0.5$。原方法和改进方法传递同一个整数20及同一个最大值1.5。

### 3.2 计数、CDF与算术码流

原codec沿每个层/通道的token方向统计符号次数，生成整数CDF。256-token块中的三个安全算术段共用该应用块CDF；每条变长子码流另外保存int32长度。

~~~text
原方法：

发送端：量化 → 统计计数 → 生成CDF → 算术编码
          │                  │           │
          ↓                  ↓           ↓
网络： [最大值、格式信息]    [CDF]    [主体码流、子流长度]
          │                  │           │
          │                  └─────┬─────┘
          │                        ↓
          │                    原算术解码
          │                        │
          └──────────┬─────────────┘
                     ↓
                  原反量化
                     ↓
                 布局恢复与拼接
~~~

原消息包含主体码流、CDF、子流长度、最大值参数、形状与量化配置。原算术编码和解码已在GPU上执行；新方案改变的是量化之后的表示与恢复方式。

### 3.3 原GPU恢复的数据路径

当前接收端依次执行：

~~~text
算术解码，输出整数符号
  → 转FP32
  → 减量化偏移
  → 除量化系数
  → 乘最大值参数
  → stack K/V
  → 整理布局、转FP16
  → 与累计KV前缀torch.cat
~~~

多个逐元素运算产生中间张量与显存读写。逐块torch.cat重复复制已恢复的前缀。permute创建视图；布局相关复制按reshape、cast和contiguous对应事件单独记录。

## 4 原方法的字节构成与优化动机

以下MB均为 $10^6$ B。源码形状给出张量字节，完整消息采用已有实测。

### 4.1 4K的两种原消息组织

| 消息组成 | 16个256-token应用块 | 4096-token整包 |
| --- | ---: | ---: |
| 原始FP16 KV张量 | 536.870912 MB | 536.870912 MB |
| 原CDF张量 | 69.206016 MB | 4.325376 MB |
| 算术子流int32长度表 | 12.582912 MB，48段 | 11.272192 MB，43段 |
| FP16最大值参数 | 0.524288 MB | 0.524288 MB |
| 完整原消息 | 151.884629 MB | 约89.365211 MB |

整包的完整字节来自质量任务20个来源的均值；分块字节来自对应计时条件。两者共同说明块长会影响元数据和主体码流，跨文件的字节/耗时换算作为规划代理使用。

### 4.2 高带宽下的动机

高带宽降低了少传若干字节的时间价值，而CDF生成、算术编解码、张量分配、设备搬运和重复拼接仍有成本。本方案固定采用定长符号表示，并将解包和恢复合并执行。

原短块格式重复发送CDF和长度表。按照第6节固定协议，新分块消息为141,034,496B，小于历史分块均值151,884,628.6B；新整包为141,033,536B，大于历史整包均值89,365,210.7B。整体耗时由第8节的计算、搬运与传输成本关系描述。

## 5 方法：混合4/5bit打包与融合恢复

### 5.1 保持原量化配置

| 数据 | 层号，从0开始 | bins | 实际符号范围 | 固定位宽 |
| --- | --- | ---: | --- | ---: |
| K | 0～9 | 32 | 0～30 | 5bit |
| K | 10～31 | 16 | 0～14 | 4bit |
| V | 0～1 | 32 | 0～30 | 5bit |
| V | 2～31 | 16 | 0～14 | 4bit |

64个平面中12个采用5bit、52个采用4bit，平均位宽：

$$
\bar b=\frac{12\times5+52\times4}{64}=4.1875\ {\rm bit/标量}.
$$

量化固定调用原torch_quant_vectorized，保留其舍入、零分组处理、符号和FP16最大值。量化函数与bins张量dtype保持原实现；打包只改变整数符号的可逆存储方式。

### 5.2 4bit：两个符号合并为一个字节

约定第一个符号占低4位，第二个占高4位：

$$
u=q_0\;|\;(q_1\ll4).
$$

解包：

$$
q_0=u\;\&\;15,\qquad q_1=(u\gg4)\;\&\;15.
$$

~~~text
符号：       2        7        1       14
各自4bit： 0010     0111     0001     1110
打包字节：     01110010          11100001
十六进制：        72                e1
~~~

四个原uint8符号由4B缩为2B。CUDA线程各处理8个符号，输出4个互不重叠的字节。

### 5.3 5bit：八个符号合并为五个字节

八个符号恰好占40bit：

$$
P=\sum_{i=0}^{7}q_i2^{5i}.
$$

约定以小端序输出低40位，第 $j$ 个字节为 $(P\gg8j)\;\&\;255$。恢复第 $i$ 个符号：

$$
q_i=(P\gg5i)\;\&\;31.
$$

例子：符号 $[1,2,3,4,5,6,7,8]$ 打包为五个十六进制字节：

~~~text
41 0c 52 cc 41
~~~

CUDA线程各处理8个符号，用uint64_t组装40位整数，按字节写出5B。所有线程的输出区间互不重叠，整个pack过程不使用原子操作。

v1固定 $D=1024$，所有合法 $N$ 都满足 $ND$ 被8整除；每平面的最后一组仍有8个符号。尾块按实际 $N$ 计算长度，pack kernel对线程组索引执行边界判断。

### 5.4 发送与恢复总图

~~~text
改进方法：

发送端：             原量化 ──→ GPU 4/5bit打包
                       │               │
                       ↓               ↓
网络：         [最大值、格式信息]   [定长打包数据]
                       │               │
                       └───────┬───────┘
                               ↓
接收端：            GPU融合解包＋反量化
                               ↓
                     直接写入预分配的最终KV
~~~

CPU socket传输的具体搬运链为：

~~~text
GPU量化/打包 → D2H到CPU → 封装与socket发送
              → CPU接收/解析 → H2D上传打包字节和最大值
              → GPU融合解包、反量化、写入最终KV
~~~

D2H、H2D、序列化和同步全部计入对应阶段。CPU接收端保留压缩字节，将解包留给GPU，避免先扩成完整uint8符号再上传。

### 5.5 融合解包、反量化和类型转换

融合kernel按最终输出布局分配线程，对每个目标元素计算源地址，提取4bit或5bit整数，执行FP32减法、除法和乘法，再直接写出FP16。

~~~cpp
// 单个输出元素的固定数值路径，包含cuda_fp16.h
float v = __fsub_rn(static_cast<float>(q), gamma);
v = __fdiv_rn(v, gamma);
v = __fmul_rn(v, __half2float(original_max));
final_kv[destination_index] = __float2half_rn(v);
~~~

正式路径不分配完整uint8解包输出、FP32恢复中间张量和独立stack输出。中间数值由kernel局部变量传递；最终只写FP16目的区间。

CUDA编译参数固定为-O3、--fmad=false、--ftz=false、--prec-div=true，不启用--use_fast_math。数值恢复以原do_dequantize的FP32运算顺序和FP16舍入结果为基准，保持FP16位模式相同。

### 5.6 预分配最终KV，直接写入token区间

请求开始时按已知完整长度分配：

~~~python
final_kv = torch.empty(
    (layers, 2, kv_heads, total_tokens, head_dim),
    dtype=torch.float16, device="cuda",
)
~~~

当前块的符号属于K/V侧 $s$、层 $l$、head $h$、块内token $t$、head内维度 $d$ 时，写入：

~~~text
final_kv[l, s, h, chunk_start + t, d]
~~~

块内符号源索引为 $i=tD+hd_{\rm head}+d$；平面号为 $p=sL+l$。因此从目标坐标直接计算打包偏移和scale位置，完成最终布局写入。

4K、16块时，现有15次累计cat产生4.21875GiB输出，逻辑读写合计8.4375GiB。正式路径在请求开始时分配连续最终KV，kernel直接写对应区间，结束后向模型交付该张量。每块恢复完成前，该块区域不交给模型读取。

8.4375GiB是逻辑复制量，不是耗时结论。profiling分别记录cat、反量化和设备复制；最终性能使用完整请求墙钟。

### 5.7 与原方法、低带宽方案的区别

| 环节 | 原算术路径 | 低带宽组合计数方案 | 本文高带宽方案 |
| --- | --- | --- | --- |
| 原量化符号、最大值 | 生成并保留 | 保持相同 | 保持相同 |
| 统计和CDF | 生成局部CDF | 由计数编号恢复原CDF | 定长表示直接确定符号边界 |
| 符号主体 | 算术码流 | 原算术码流 | 混合4/5bit数据 |
| 概率元数据 | 原CDF | 组合编号 | 由位宽配置解包 |
| 接收端 | 算术解码后反量化 | 重建CDF后执行原解码 | 解包和反量化融合 |
| 主要取舍 | 压缩率与编解码成本 | 少传元数据、增加编号处理 | 固定整包多传51.67MB，改用位操作和融合恢复 |

位打包改变符号表示；预分配和融合反量化改变GPU恢复路径，这两项分别作用于数据表示与执行方式。低带宽CDF组合编码保持独立，本文采用mixed45表示和融合恢复。

## 6 固定消息协议与收发流程

### 6.1 M45F v1：64字节消息头

整个wire由“64B头＋原FP16最大值＋混合位宽符号”组成。新方法直接向transfer_chunks交付该wire，不再追加CGV3头、JSON、pickle或torch.save容器。原方法的消息格式保持原样。

所有多字节字段及FP16最大值采用小端序。Python结构定义固定为：

~~~python
HEADER = struct.Struct("<4sHHIHHHHIIIIIQQQ")
assert HEADER.size == 64
~~~

| 字节偏移 | 长度 | 字段 | 固定值或定义 |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | ASCII M45F |
| 4 | 2 | version | uint16，1 |
| 6 | 2 | header_bytes | uint16，64 |
| 8 | 4 | flags | uint32，0 |
| 12 | 2 | layers | uint16，32 |
| 14 | 2 | kv_heads | uint16，8 |
| 16 | 2 | head_dim | uint16，128 |
| 18 | 2 | schedule | uint16，2，表示原cachegen2配置 |
| 20 | 4 | token_start | uint32，当前块起点 |
| 24 | 4 | token_count | uint32，当前块实际token数N |
| 28 | 4 | total_tokens | uint32，完整上下文长度T |
| 32 | 4 | chunk_index | uint32，从0开始 |
| 36 | 4 | chunk_count | uint32，整个请求的消息数A |
| 40 | 8 | scale_bytes | uint64，2LN×2 |
| 48 | 8 | symbol_bytes | uint64，各平面定长字节之和 |
| 56 | 8 | reserved | uint64，0 |

正文顺序固定为：

~~~text
[64B消息头]
[K层0…31、V层0…31的FP16最大值，各平面按token升序]
[平面0符号 | 平面1符号 | … | 平面63符号]
~~~

平面内符号按token、channel顺序展平。位宽和gamma从固定schedule推导，协议不携带CDF、逐通道长度、逐平面偏移表及额外对齐字节。

接收端验证magic、版本、固定几何、schedule、flags/reserved零值、正文长度和token范围。分块模式要求start=index×256、count=ceil(T/256)、N=min(256,T−start)；整包要求index=0、count=1、start=0、N=T。消息次序与当前接收前缀不符时抛出ValueError，终止该请求，不返回部分KV。

### 6.2 符号偏移与GPU解包

平面p的符号字节数、相对符号区起点：

$$
s_p=ND b_p/8,\qquad o_p=\sum_{j<p}s_j.
$$

平面内第i个符号的bit起点是ib_p。读取字节位置为o_p+floor(ib_p/8)，移位量为ib_p mod 8。kernel将当前字节与下一字节组合为uint16，移位后与2^{b_p}−1按位与；下一字节超出当前平面时取0。

CPU按固定schedule和N计算64个平面的偏移，GPU直接使用同一计算规则。4bit平面的单平面长度为512N字节，5bit平面为640N字节。

### 6.3 固定缓冲区与同步规则

发送端每次完整上下文编码分配一组最大应用块大小的GPU正文缓冲区和pinned CPU正文缓冲区，逐块复用。原量化的K/V最大值逐位复制到GPU正文前部，pack kernel直接写正文符号区；随后一次阻塞D2H复制正文，CPU生成64B头并创建不可变wire bytes。当前wire构造完成后复用缓冲区。缓冲区分配、复制和bytes构造均计入编码时间。

接收端在请求计时开始后分配最终FP16 KV、最大应用块大小的pinned CPU正文缓冲区及GPU正文缓冲区。解析头后将正文复制到pinned缓冲区，一次阻塞H2D后启动恢复kernel；当前默认CUDA stream完成后返回restore回调。下个块复用同一组正文缓冲区。

pipeline固定保留原CPU socket接收与GPU恢复重叠，queue_capacity=1。GPU复制和计算统一使用主线程默认stream。本版本不增加多stream搬运流水线。

### 6.4 固定CUDA kernel组织

| kernel | 用途 | 固定线程映射 |
| --- | --- | --- |
| pack_mixed45 | 量化后的符号打包 | block=256线程；grid.y=64平面；每线程处理8个输入符号并写4B或5B |
| unpack_dequant_write | 融合解包、反量化和最终写入 | block=256；grid.y=64；每线程解包、反量化并写一个最终FP16元素 |

pack的grid.x=ceil((ND/8)/256)，融合恢复kernel的grid.x=ceil(ND/256)。所有kernel对线程索引执行边界判断，使用输入所属设备的CUDAGuard及c10::cuda::getCurrentCUDAStream(device).stream()。构建目标固定TORCH_CUDA_ARCH_LIST=8.9。

融合恢复kernel按输出顺序映射线程：对平面内线性输出索引u，d=u mod 128、t=floor(u/128) mod N、h=floor(u/(128N))，再从i=tD+128h+d取得符号。目的索引按[l,s,h,start+t,d]计算。4/5bit同处一次kernel launch，按平面选择固定分支。

## 7 固定协议的完整传输大小

### 7.1 符号与最大值

4K完整上下文的标量数：

$$
V=2LTD=268,435,456.
$$

符号数据：

$$
S_{\rm symbol}=TD(12\times5+52\times4)/8
=140,509,184\ {\rm B}.
$$

FP16最大值数据：

$$
S_{\rm max}=2LT\times2=524,288\ {\rm B}.
$$

每个应用消息固定增加64B。完整发送字节数为：

$$
S_{\rm fix}=141,033,472+64A\ {\rm B}.
$$

这是协议确定值，包含所有应用消息字节。操作系统socket协议开销不计入应用wire，与原实验统计边界一致。

### 7.2 分块与整包的精确值

| 组织 | 每消息符号B | 每消息最大值B | 每消息头B | 每消息wire B | 完整4K wire B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16个256-token块 | 8,781,824 | 32,768 | 64 | 8,814,656 | 141,034,496 |
| 1个4096-token整包 | 140,509,184 | 524,288 | 64 | 141,033,536 | 141,033,536 |

分块多出的960B完全来自额外15个消息头。尾块使用实际N，公式逐块相加。

### 7.3 与已有消息的比较

| 表示 | 完整4K应用消息字节 | 数值性质 |
| --- | ---: | --- |
| 新mixed45，256-token分块 | 141,034,496 B | M45F v1协议确定值 |
| 新mixed45，整包 | 141,033,536 B | M45F v1协议确定值 |
| 原CacheGen2，256-token分块 | 151,884,628.6 B | 历史实测均值 |
| 原CacheGen2，整包 | 89,365,210.7 B | 历史质量任务均值 |

新分块消息相对历史原分块均值减少10,850,132.6B，约7.14%；新整包增加51,668,325.3B。每次实验再次记录实际len(wire)并与协议公式核对。

原始FP16张量为536,870,912B；原量化符号按uint8保存并加最大值为268,959,744B。这两项只用于解释数据组成，不进入新协议。

### 7.4 固定代码来源

量化和最大值计算直接调用cachegen_encoder.py的torch_quant_vectorized。mini_repro/experiment_common.py的uniform4作为既有系统对照，保持其原实现。experiments/mechanisms.py中的CPU混合位宽实现用于交叉核对整数位流，不进入GPU性能路径。

## 8 历史实测与成本边界

### 8.1 固定计时边界

- 首次总时间：原始KV已在GPU就绪后开始，顺序计入一次完整编码、传输、恢复和首token ID在CPU可用的时间。
- 复用TTFT：使用已编码消息，从请求开始、分配接收缓冲区之前计时，到首token ID在CPU可用。
- 源端prefill：从文本生成原始KV，历史4K注记约774ms，单独报告。
- 编译、模型加载和进程级初始化：属于冷启动，不计入稳态请求处理。
- 每个上下文的编码缓冲区分配计入编码；每个请求的接收和最终KV缓冲区分配计入请求。首次请求仅计当前表示的完整编码。

首次总时间对应一次完整执行的墙钟。缓存复用直接使用已准备消息，每次请求建立独立接收状态和最终KV。

### 8.2 成本关系

串行关系定义为：

$$
T_{\rm first}=E+8S/\beta+R+U.
$$

E为完整编码、R为完整接收恢复、U为其余请求工作。令DeltaS=S_fix−S_ar、DeltaU=U_fix−U_ar，则净节省为：

$$
g_{\rm first}=(E_{\rm ar}-E_{\rm fix})+(R_{\rm ar}-R_{\rm fix})
-\frac{8\Delta S}{\beta}-\Delta U.
$$

复用TTFT去掉编码项；复用r次的摊销编码节省为(E_ar−E_fix)/r。成本边界表取DeltaU=0；完整请求时间包含所有实际剩余工作。

当DeltaS>0、净计算节省C_save>0时，盈亏平衡带宽为：

$$
\beta^*=\frac{8\Delta S}{C_{\rm save}}.
$$

C_save从相同计时范围取编码/恢复及剩余处理的净差。带宽采用实际goodput。pipeline各阶段有重叠，阶段耗时仅用于分析，总时间取实际墙钟。

### 8.3 历史高带宽实测参照

| 方法与模式 | 编码ms | 在线TTFT ms | 编码加一次请求ms |
| --- | ---: | ---: | ---: |
| CacheGen2，256-token串行 | 215.8 | 510.5，配对均值 | 726.3 |
| CacheGen2，整包 | 230.2 | 463.5，中位数 | 693.7，历史规划代理 |
| CacheGen2，256-token流水线 | 215.8 | 445.5，中位数 | 661.4，历史规划代理 |
| 现有uniform4，256-token流水线 | 183.3 | 390.2，中位数 | 573.5，历史规划代理 |

历史代理为编码均值加在线中位数，用于成本分析；单次完整请求的总时间对应实际墙钟。

10Gbps为本地socket目标。历史4K CacheGen2实测goodput分别为整包约4.95Gbps、分块串行约6.66Gbps、分块流水线约5.37Gbps。

### 8.4 数值成本边界

按固定新整包141,033,536B和历史原整包89,365,210.7B，在有效吞吐保持不变的预算条件下，传输144.8ms对应新传输228.5ms，增加83.7ms。历史整包在线中位数减去传输/恢复均值的预算残差为153.1ms。

| 总时间条件 | 对应的新处理成本上限 |
| --- | ---: |
| 首次总时间比CG2整包下降20% | 编码＋恢复173.4ms |
| 首次代理比历史uniform4流水线下降20% | 编码＋恢复77.2ms |
| 在线低于历史CG2整包463.5ms | 恢复81.9ms |
| 在线低于历史uniform4流水线390.2ms | 恢复8.6ms |

表中上限由历史整包数据、吞吐保持不变与DeltaU=0推导，描述新增传输开销能够容纳的编码和恢复成本。

### 8.5 与低带宽方案的关系

低带宽方法见 [低带宽CDF组合编码优化方案](低带宽CDF组合编码优化方案.md)。低带宽方案压缩概率元数据并保留原算术码流；高带宽方法采用mixed45位打包和融合恢复，以更简单的处理换取带宽。

原量化与恢复逐位一致是本方案的质量控制条件。cachegen1在历史4K质量门槛未通过，不作为精度等价的提速路径。

## 9 来源、复算与文档入口

### 9.1 固定提交源码与正式结果

- [量化、CDF、算术编码和collect_bytes](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/lmcache/storage_backend/serde/cachegen_encoder.py)。
- [原解码、FP32反量化和布局转换](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/lmcache/storage_backend/serde/cachegen_decoder.py)。
- [量化配置与安全内部段](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/lmcache/storage_backend/serde/cachegen_basics.py)。
- [应用分块、消息封装和累计cat](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/experiments/chunks.py)。
- [已有GPU uniform打包](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/mini_repro/experiment_common.py#L68-L109)。
- [混合位宽CPU消融](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/experiments/mechanisms.py)。
- [编码CUDA launch](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/third_party/torchac_cuda/torchac_kernel_enc_new.cu#L362-L371)、[解码launch](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/third_party/torchac_cuda/torchac_kernel_dec_new.cu#L400-L409)、[CDF launch](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/LMCache/third_party/torchac_cuda/cal_cdf.cu#L80-L87)。
- [正式TTFT汇总](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/results/mistral/larger/v3-larger-stable-20261025/ttft.csv)、[完整消息与质量](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/results/mistral/larger/v3-larger-stable-20261025/quality.csv)、[机制消融](https://github.com/ZClucky-123/CacheGen/blob/d6b6626014c30b64011e4348c02510fdfbdb7204/results/mistral/larger/v3-larger-stable-20261025/ablation.csv)。
- [PyTorch 2.3 CUDAExtension](https://docs.pytorch.org/docs/2.3/cpp_extension.html)。
- [CUDA FP32显式舍入运算](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)、[FP16转换](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html)。

### 9.2 仓库证据与复算

历史结果见[实测报告索引](../../reports/README.md)、[公共框架历史](../../docs/reproduction/HISTORY.md)及[环境说明](../../docs/reproduction/MISTRAL_ENVIRONMENT.md)。独立整数和线格式验证见[联合codec测试](../../tests/test_joint_codec.py)。

联合请求的准备策略、三编码候选、缓存、调度和计时以[整体方案v2.2](CacheGen整体优化与代码修改方案.md)为准。单路径分析中的条件推算保留其假设；目标GPU结果由独立运行产物记录。

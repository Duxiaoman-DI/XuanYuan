
# XuanYuan-70B

## 概览

XuanYuan-70B 金融大模型是基于Llama2-70b模型进行增量预训练得到的一个基座模型，预期在中文能力和金融能力方面，相比原始模型得到较大提升，技术优化点包括：

（1）**数据质量**

- 我们设计了一套数据清洗流水线，精心准备了各类通用数据（互联网网页、百科、论坛、社交媒体、问答等）以及金融相关数据（金融资讯、公司公告、金融百科、金融书籍、证书试题等）高质量数据
- 中英数据：首先llama2的英文能力足够优秀，所以为了保证英文能力不降，我们扩充词表之后，使用高质量的中英语料进行增量预训练，其中中英配比为3:1；
- 通用金融数据：为了提升模型在金融能力上效果，预训练过程中通用语料与金融预料比例为9:1，且随着训练进行，逐步提升金融语料的占比。

（2）**模型训练**

- 训练效率：我们采取了一系列的加速优化策略， 包括对底层数据加载和分布式训练框架的多处优化，使用flash attention2替代self-attention模块，使用基于CPP CUDA的Fused算子替代原始llama的python实现等
- 上下文长度：基于上述的优化方式，同时考虑到金融场景长上下文情景较多，我们能够在预训练阶段把llama2原始上下文4k的长度扩展到8k和16k；

我们在100台8卡A800(80G)的GPU集群中，训练情况如下：

| 模型         | 上下文长度 | 吞吐量           | 显卡利用 |
| ------------ | ---------- | ---------------- | -------- |
| XuanYuan-70B | 8192       | 340 tokens/s/gpu | 190TFOPS |

备注：（1）训练没有开梯度累计；（2）原始llama2-70b在4k上下文长度下的的吞吐量为323 tokens/s/gpu，说明我们的训练效率达到当前领先水平。

## 数据质量

数据质量是影响大模型的训练效果的最关键因素，我们针对各类数据设计了一套通用的数据清洗流水线，主要包含如下的数据预处理：

- 文本抽取：原始数据中存在大量的互联网页面 和 PDF数据（如研报、公告、书籍等），我们针对这两类源数据做了详细的内容抽取工作，HTML标签移除、PDF内容深度解析等技术，特殊格式如表格使用markdown存储，数学公式使用latex等，保证文本内容尽可能全而准的抽取出来。
- 数据清洗：基于抽取的文本内容，我们设计了更深层次的数据清洗模块，包括词级别、句子级别、文档级别的数据清洗，根据不同数据集的分布特点，制定几十类不同的清洗规则和过滤阈值，这一步实现了语法、词法、基本语义的清洗。
- 数据去重：这一步我们使用MinHashLSH算法，采取单类别的局部去重+全类别的全局去重两阶段策略，来去除重复数据。
- 质量筛选：为了进一步过滤有毒、有害的信息，我们通过人工标注+关键词抽样的构造方式，形成了一批有害样本，涵盖暴力、色情、歧视、反动、博彩、价值观错误等多种有害信息，基于上述样本，我们训练了一个文档质量分类模型，来对语料进行质量过滤。

数据清洗完之后，下一步预训练数据的配比。首先我们认为Llama2-70b本身的中文理解能力是足够的，而中文知识比较匮乏，因此我们在增量预训练着重训练知识类数据（比如百科、书籍、教材、试题、论文等），之后是综合类数据（网页文本、新闻资讯、社交媒体类等），同时为了保持英文能力，我们在预训练中加入了部分英文数据集SimPajama。

同时我们积累了海量的金融语料，涵盖了研报，财报，公告，资讯，论坛，金融百科、金融书籍试题等，因此为了提升模型的金融理解能力，我们也在增量预训练过程中，加入了高质量的金融语料数据。

综合中英、通用金融来说，我们的增量预训练过程中数据整体配比介绍如下：

- 中英数据配比大致在3:1的比例，英文数据从SimPajama中类别中进行抽样得到；中文数据则包含通用和金融领域数据。
- 分阶段调整数据配比：前期中文数据以知识类为主，预期让模型学到更多中文知识，随着训练的进行，我们逐步加大其他类型的中文数据，来增加多样性。 同时金融数据的比例也是随着训练的进行逐步提升，最初设置为1:9。训练的最终阶段，金融与通用数据的比例大概为1:4左右



## 模型训练

本部分主要包含我们的在模型训练层面的细节，主要目的是（1）如何能够快速高效的训练llama2-70b，（2）提升模型的上下文长度。

### **（1）增量预训练**

考虑到原生llama2的中文能力较弱，需要进行中文增强。首先llama2的原始词表包含32000个token，其中中文字符很少，导致大部分的中文内容，需要解码成多个unicode字符，因此同样的文本内容，编码序列会变长，解码速度会降低。

因此首先需要扩充中文词表，将常见的中文字符加入到词表中。然后基于新词表进行中文增量预训练。 

首先是词表扩充，一般有基于词粒度和字粒度的扩充，各有优劣：

- 词粒度：词表扩充幅度大，一般需要20k以上，对原始模型形成破坏更多，且对字粒度的任务不友好；不过解码效率会更高，词表压缩率高，同样的内容，tokenize之后的序列更短。
- 字粒度：中文字粒度数量少，一般需要5-10k，相对来说，对原始模型破坏较少。 不过相比词粒度，压缩率会偏低，解码效率也偏低。

综合考虑llama2-70b的参数量，保险起见选择了字粒度扩充，一共新加入约7k的字符，共达到39k左右的词表大小。

之后是增量预训练策略，常见也有多种方法：

- 扩充之后只更新模型开始的embeding和最后的解码线性层，其他参数保持不更新
- 扩充之后全量更新所有模块的参数。

我们选择两者结合训练方法：

- 一阶段使用较少的数据仅仅更新模型的词表特征以及解码线性层，模型其他参数保持不变，这一阶段目的是让模型适应新加入的token，纠正原始模型的中文解码方式。本身llama2的中文理解能力已经足够，只是中文知识缺乏。因此这一步的目的是纠正中文的解码方式。
- 二阶段则使用大量的中英数据，对模型进行全参数更新，目的是保持英文能力不下降的同时，中文能力通过参数更新获得提升

### （2）模型训练加速

本部分主要介绍关于训练速度优化方面的工作，从如下三个层面展开：

**（1）数据加载优化**

我们将文本数据提前做tokenize，预处理存储为二进制格式，预训练阶段直接加载二进制索引，可以大幅提升数据加载速度，降内存占用。相比直接加载文本数据，更加高效，单机内存可以轻松快速加载TB级别的数据。

**（2）模型结构优化**

本部分介绍在模型本身上的优化措施，具体包括：

- 使用Flash Attention2 替代self-attention, 相比原始实现，Flash Attention2能够大幅降低显存占用，提升训练速度。
- 算子融合操作：使用一些cuda融合算子来替代原始Llama2的python实现，可以提升训练速度

备注：由于模型上下文长度较长，我们也修复了llama2模型中因为bfloat 16导致rotary embdding的参数冲突的问题。



**（3）分布式训练底层优化**

我们使用DeepSpeed来作为基本的分布式训练框架，并且从多个方面来进行了优化增强：

- 优化分布式训练网络配置，提升多机多卡分布式通信效率，包括RoCE高性能网络优化及网络拓扑感知，提升网络吞吐及降低时延，实现多机加速比达到90%以上（如40台吞吐351tokens/s，100台吞吐340tokens/s/gpu，340/351=96.8%)
- 通过故障自愈和弹性扩缩容策略，实现快速发现故障（GPU硬件，机器，网络等）及训练恢复，提升训练稳定性和健壮性。降低训练中断时间，70B模型的故障平均恢复时间在1小时内。



# XuanYuan-70B-Chat

基于上述的XuanYuan-70B基座模型，我们进行了详细的指令微调，基座使模型具备对话和遵循人类指令的能力。

## 数据与训练

我们采取了两阶段的指令微调，具体来说：

- 第一阶段：使用开源的大量的指令数据对基座模型来进行训练，这一部分我们收集了约10M条开源的多语种指令微调数据，并行清洗与深度过滤。这一阶段的目的是为了覆盖指令的多样性，提升模型指令遵循能力。
- 第二阶段：使用自研的高质量的指令数据来继续进行指令微调训练。这一阶段，我们精心自研约20万条通用+金融的指令微调数据，其中大部分数据均做了校验、改写来保证质量。 这一阶段是能够更加使得模型根据不同的需求和侧重来进行最后的训练。

我们自研的指令微调数据预期模型能够在通用对话能力保留的同时，更加侧重金融领域的问答。具体来说，通用指令数据分为以下几个大类：常识百科、代码编程、逻辑推理、数学计算、创意生成、安全无害、摘要提取、翻译等。其中每一大类下又设计了多个子类，来尽可能保证指令数据的多样性和丰富度。

对于金融领域的指令数据，我们进行了更加详细的子类划分，来覆盖金融经济的各个领域。在训练阶段，我们采取的配比为：通用指令数据与金融指令数据配比为4：1。



在训练过程中，我们同样保持8k的上下文长度，未采取外推的措施来提升上下文。后续我们将继续在预训练阶段来提升上下文长度。 同时训练数据中的question-answer pair，我们仅对answer部分计算损失。





## 模型量化

为了降低用户在本地使用XuanYuan的成本，降低显存需求，我们提供了离线的量化模型以及量化方法。您可以选择使用我们的8bit量化模型和4bit量化模型。

### 8bit离线量化模型

在8bit量化算法上，我们使用目前社区广泛使用的[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)库。该库包含LLM.int8()量化算法的实现以及一系列量化的工具，
同时该方法已在transformers库里做了集成，使用较为容易。经过我们的测试，8bit量化可以近乎无损。

**（1）直接使用**

您可以直接下载我们量化好的8bit模型：huggingface链接

首先，如果您还未安装bitsandbytes：

```bash
pip install bitsandbytes
```

然后，使用类似下面的脚本推理出答案:

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name_or_path = "Duxiaoman-DI/XuanYuan-70B-Chat"

tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
model.eval()
system_message = "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与 与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
seps = [" ", "</s>"]
roles = ["Human", "Assistant"]

content = "介绍下你自己"
prompt = system_message + seps[0] + roles[0] + ": " + content + seps[0] + roles[1] + ":"
print(f"输入: {content}")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"输出: {outputs}")
```



**（2）8bit量化**

如果您想8bit量化自己训练好的模型，您可以在安装bitsandbytes后使用如下的脚本：

```python
import torch
from transformers import AutoModelForCausalLM

model_id = "/your/model/path"
quant8_saved_dir = "/8bit/model/saved/path"
# 加载并量化你的原始模型
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto", trust_remote_code=True)
# 保存量化后的模型
model.save_pretrained(quant8_saved_dir)
```



### 4bit离线量化模型

在4bit量化算法上，我们使用[auto-gptq](https://github.com/PanQiWei/AutoGPTQ)工具。该库实现的GPTQ算法是目前4bit量化最受欢迎的方法，
同时该方法在transformers库和optimum库里做了集成，使用较为容易。

**（1） 直接使用**

您可以直接下载我们量化好的4bit模型。

首先，如果您还未安装auto-gptq和optimum：

```bash
pip install auto-gptq optimum
```

然后，使用上面同样的脚本推理出答案。



**（2）4bit量化脚本**

如果您想4bit量化自己训练好的模型，您可以在安装auto-gptq和optimum后使用如下的脚本：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "/your/model/path"
quant4_saved_dir = "/4bit/model/saved/path"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# 您需要使用自己的数据集来做校准
dataset = ["The greatest test of courage on earth is to bear defeat without losing heart..."]  
quantization_config = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)
# 加载并量化初始模型
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
# 保存量化后的模型
model.save_pretrained(quant4_saved_dir)
```
### 量化效果

下表给出了不同模型所需显存，以及在三个评测基准上CEVAL，CMMLU和MMLU上效果：

| 模型                   | 显存 | CEVAL | CMMLU | MMLU |
| ---------------------- | ---- | ----- | ----- | ---- |
| XuanYuan-70B-Chat      | 129G | 62.15 | 60.41 | 65.3 |
| XuanYuan-70B-Chat-8bit | 65G  | 62.25 | 59.99 | 65.0 |
| XuanYuan-70B-Chat-4bit | 35G  | 60.94 | 58.76 | 63.0 |

8bit的量化模型相原始float16的模型，效果近乎无损，4bit的量化模型，大概下降2个点左右。不过存储要求降低很多。
此外，我们也对量化版本的Chat模型进行对话人工评测，结论与评测基准类似。

需要注意的是，Chat模型相比Base模型在榜单指标上有所下降，我们认为是符合预期的，SFT阶段我们更加重视指令遵循能力和内容生成方面的能力。


# 总结

本次我们开源的XuanYuan-70B系列模型弥补了原始Llama2-70B的中文较弱的缺点，此外为了更好服务金融场景，我们加入了更多的金融领域数据，提升金融理解的能力。

此外我们也会持续分享数据和模型训练方面的实践细节，也欢迎大家加入我们主页的微信群进行交流。

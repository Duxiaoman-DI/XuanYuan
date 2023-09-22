# 概览：
XuanYuan-70B-base 金融大模型是基于Llama2-70b模型进行增量预训练得到，预期在中文能力和金融能力方面，相比原始模型得到较大提升，技术优化点包括：

（1）**数据准备**

- 中英数据：首先llama2的英文能力足够优秀，所以为了保证英文能力不降，我们使用扩充词表之后，使用高质量的中英语料进行增量预训练，其中中英配比为3:1；
- 通用金融数据：为了提升模型在金融能力上效果，我们设计了一套数据清洗流水线，精心准备了金融资讯、公司公告、金融百科、金融书籍、证书试题等高质量数据；预训练过程中通用语料与金融预料比例为9:1，且随着训练进行，逐步提升金融语料的占比。

（2）**模型训练**
- 训练效率：我们采取了一系列的加速优化策略， 包括对底层数据加载和分布式训练框架的多处优化，使用flash attention2替代self-attention模块，使用基于CPP CUDA的Fused算子替代原始llama的python实现等
- 上下文长度：基于上述的优化方式，同时考虑到金融场景长上下文情景较多，我们能够在预训练阶段把llama2原始上下文4k的长度扩展到8k和16k；

我们在100台8卡A800(80G)的GPU集群中，训练情况如下：
|模型|上下文长度|吞吐量|显卡利用|
|--|--|--|--|
|XuanYuan-70B|	8192|	340 tokens/s/gpu|	190TFOPS|

备注：（1）上述训练没有开梯度累计；（2）原始llama2-70b在4k上下文长度下的的吞吐量为323 tokens/s/gpu，说明我们的训练效率达到当前领先水平。
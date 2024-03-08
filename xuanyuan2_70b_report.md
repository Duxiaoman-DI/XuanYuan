# XuanYuan2-70B

新一代的XuanYuan2-70B系列金融大模型在XuanYuan-70B系列模型的基础上使用了更多高质量金融和通用语料进行继续训练和指令微调，同时使用基于人类反馈的强化训练进一步对齐模型表现和人类偏好，使得模型的通用性、安全性和金融能力得到了进一步提高。此外，XuanYuan2-70B系列模型支持的上下文长度达到16k，能够更好地应用于长上下文的业务场景。

XuanYuan2-70B系列包含以下模型：XuanYuan2-70B-Base、XuanYuan2-70B-Chat、XuanYuan2-70B-Chat-8Bit、XuanYuan2-70B-Chat-4Bit。XuanYuan2-70B-Base是在XuanYuan-70B-Base基础上，使用更多高质量的通用和金融语料进行继续预训练得到，同时支持的上下文长度达到16k。XuanYuan2-70B-Chat是在XuanYuan2-70B-Base基础上，使用数量更多、多样性更强、质量更高的语料进行指令微调和基于人类反馈的强化对齐而得到。XuanYuan2-70B-Chat-8Bit、XuanYuan2-70B-Chat-4Bit分别是XuanYuan2-70B-Chat的8-Bit和4-Bit量化版本，可在配置更低的硬件资源上进行部署使用。

接下来，我们介绍XuanYuan2-70B系列模型的技术细节，包括预训练、指令微调和强化对齐。

## 预训练

在XuanYuan-70B基座模型的基础上，我们持续加入更高质量的预训练数据进行训练。同时为了兼顾训练效率和长文本建模，提出了一种**数据分桶的动态预训练方法**。

具体来说，根据不同的数据类型，将训练数据的长度动态的按照2k、4k、8k、16k等4种上下文长度来进行拼接，我们设计一种基于贪心思想的分桶策略，实现长文本的数据(如书籍、论文)更多落在16k的数据桶中、短文本数据（如新闻）更多在较短的桶中。在实际训练过程可以每次随机选择一种数据桶来进行训练。相比先用短上下文训练，在用长上下文训练的方式，分桶训练更可以保证每一批次训练数据的随机性和多样性，可以减少短文本被大量拼接到一条数据的情况。此外由于短数据训练速度更快，这种分桶策略能够更自然的提升训练速度。

以少量的CommonCrawl数据为例，使用上述分桶策略之后，各类数据以及对应训练速度如下所示，

| 上下文长度 | 数据量 | 训练速度    |
| ---------- | ------ | ----------- |
| 2k         | 23M    | 370tokens/s |
| 4k         | 9M     | 350tokens/s |
| 8k         | 2M     | 340tokens/s |
| 16k        | 0.6M   | 305tokens/s |

基于上述的数据分桶方式，我们在第一代XuanYuan-70B-Base模型的基础上，又额外训练了300B tokens得到XuanYuan2-70B-Base模型，模型的中文理解、金融知识等指标评测均达到不同幅度的提升。

## 指令微调

与XuanYuan-70B-Chat一致，我们基于上述的XuanYuan2-70B-Base模型，重新利用更多高质量的指令微调数据来进行指令对齐，主要提升的方向是通用与金融类型的指令数据质量和多样性，具体的数据类型如下：

<img src=resources/data_category.png>

使用多种SFT数据产生方式（self-instruct, self-qa等），最终得到中文SFT约40万条通用数据+10万条金融数据来进行指令微调，在训练阶段，同样仅对answer部分计算损失。

经指令微调后，我们将得到的模型称为XuanYuan2-70B-SFT。

## 强化对齐

在XuanYuan2-70B-SFT模型基础上，我们进行了基于人类反馈的强化训练（Reinforcement learning with human feedback，RLHF），得到了XuanYuan2-70B-Chat模型。经过RLHF训练，我们进一步对齐了模型与人类的偏好，使模型表现能更符合人类需求。通过人工评估，经强化训练后，相比XuanYuan2-70B-SFT模型，XuanYuan2-70B-Chat模型在通用性、安全性、金融领域内的表现有了较明显的提升。

在训练中，我们使用了和XuanYuan-6B RLHF训练相同的数据构建方式和模型训练方法，具体可参考XuanYuan-6B的技术文档：[Report](xuanyuan_6b_report.md)。此处我们仅介绍和XuanYuan-6B RLHF训练有区别之处。

### 偏好数据构建

在构建偏好数据时，我们采用XuanYuan2-70B-SFT模型来产生response，在此基础上进行人工偏好标注。

### RM架构

我们使用XuanYuan-13B-Chat作为reward model（RM）的基本架构，之后去掉最后的LM_head layer，并将其替换为value head layer。Value head layer为一个线性层，输入是的XuanYuan-13B-Chat次顶层的特征，输出为一个一维的reward分数。我们将value head layer随机进行了初始化。在RM训练过程中，value head layer和底层XuanYuan-13B-Chat（不包括LM_head layer）联合进行训练。

### RM训练数据

在RM训练过程中，我们不仅使用了XuanYuan2-70B-SFT产生的response构建的偏好数据，也使用了额外的XuanYuan-13B-Chat产生的response构建的偏好数据。和XuanYuan-6B模型的RLHF训练相比，此处我们使用了13B的模型作为RM。由于RM尺寸增加，如果采用的训练数据量过少，模型容易陷入过拟合。因此加入一定量的额外数据对提升RM的泛化性是有帮助的，这样也更有利于后续的强化训练。

### 强化训练模型配置

强化训练需要加载4个模型，即actor model、reference model、critic model、reward model。actor model和reference model均使用XuanYuan2-70B-SFT，critic model和reward model均使用XuanYuan-13B-Chat作为基本架构。训练过程中actor model和critic model需要进行更新，而reference model和reward model则保持不变。

### 强化训练数据

如上文所述，在进行RM训练时，我们引入了额外的偏好数据，因此这部分额外偏好数据的prompt也被引入了强化训练集。对这部分prompt，我们也按照1:1的比例增加了新的prompt。在引入偏好数据的prompt和增加新prompt时，我们均对整体的prompt集合进行了去重。

### RLHF模型评估

我们采用了和XuanYuan-6B RLHF模型相同的人工评估方式，对比对象是RLHF训练前的模型，即XuanYuan2-70B-SFT。

下图展示了在通用性（安全性在评估时被纳入了通用性）、金融垂类两个领域的综合评估结果。在通用性上，相比XuanYuan2-70B-SFT，经强化训练后，XuanYuan2-70B-Chat的胜和率达到了79.29%。在金融领域，XuanYuan2-70B-Chat的胜和率达到了73.20%。这表明经强化训练后，在通用性和金融领域，模型的能力均得到了一定程度的提升。

<img src=resources/eval_all.png width=80%>

下图展示了通用性各细分领域的评估结果。从图中可以看出，在大多数细分领域，RLHF版本相比SFT版本均有了明显提升。特别地，在安全性、日常对话、数学计算等子领域，RLHF带来的提升更为明显。而在翻译、信息摘要、内容创作等子领域，RLHF并未带来预期的效果提升。在这些子领域中，我们需要进一步细化偏好标注标准，把控验收环节，提升偏好数据的质量。

<img src=resources/eval_general.png>

下图展示了金融各细分领域的评估结果。从图中可以看出，在金融客服对话、金融内容创作、金融信息摘要子领域，RLHF版本相比SFT版本有较明显的性能提升。而在金融知识理解、金融业务分析、金融计算子领域，RLHF并未带来预期的效果提升。对这些细分领域，我们需要补充更多的领域数据，同时聘请具备金融背景的专业人员进行偏好标注，以进一步提升对齐效果。

<img src=resources/eval_finance.png>
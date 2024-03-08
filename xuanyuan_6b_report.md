# XuanYuan-6B

## 概述

XuanYuan-6B系列金融大模型，包含融合了大量中英文语料增量预训练的底座模型XuanYuan-6B-Base，利用高质量指令数据和强化学习进行对齐的chat模型XuanYuan-6B-Chat以及chat模型的量化版本XuanYuan-6B-Chat-8Bit和XuanYuan-6B-Chat-4Bit。XuanYuan-6B系列模型的发布标志着智能金融领域的一次重要突破，它在日常对话、语言理解、知识应用、内容创作、信息摘要等方面可以与70B级别的模型相媲美。在模型训练前，我们收集了大量且种类丰富的预训练语料，同时对预训练数据进行多重处理来保证数据质量。在模型训练中，通过动态调整不同语种和领域知识的比例，注入丰富的专业金融语料，并灵活应用之前提出的Self-QA和混合训练方法，我们显著提升了模型在对话中的性能表现。此外，我们对模型进行了基于人类反馈的强化训练，进一步对齐了模型表现和人类偏好，提高了模型的实际体验。随着XuanYuan-6B的发布，金融领域的智能化水平将迈向一个新的台阶。它将为金融机构、投资者、研究人员等提供更准确、及时的信息，助力他们做出更明智的决策。同时，XuanYuan-6B还将成为金融教育和推广的有力工具，向广大民众提供可靠的金融知识和建议。我们将持续不断地改进和优化这个模型，以满足不断发展的金融领域需求。通过持续学习和改进，XuanYuan-6B将成为一个真正可靠且强大的金融智能伙伴，为用户带来更多惊喜和价值。

## 预训练

预训练通过无监督学习的方式，在大规模文本数据集上对模型进行训练。预训练的目标是使模型具备对语言和知识的广泛理解能力，使其能够学习词语、句子甚至整个文档的丰富表示。在预训练过程中，模型会接触到各种不同的句子和上下文，使其能够捕捉数据中存在的模式、语义和句法结构。这促使模型学习上下文关系，并捕捉文本的潜在含义。

预训练的训练数据通常来自于各种来源，如书籍、网站和其他公开可用的文本。数据的广泛性确保模型从各种主题和领域中学习，使其能够理解和生成各种上下文的文本。一旦预训练完成，模型就会获得对语言和知识的一般理解。然而，它仍然需要进一步的微调，以更好地适应特定任务，遵循人类的指令，符合人类的偏好。这个微调过程包括在特定的监督任务或指令数据上对模型进行训练，使其在各种领域（如问答、翻译、摘要或对话生成）中具备专业能力。

### 数据

数据是预训练得以成功进行的关键因素。一般而言，预训练数据需满足以下三方面条件：

* 量级大：模型参数量较大，如果预训练数据量级小，容易造成模型过拟合。同时只有足够量级的数据，才可覆盖不同语言结构与知识，模型才能从中学习到足够的语言和知识能力，具备更好的泛化性。

* 多样性强：单一来源或单一领域的数据限制了模型的学习范围，进而影响模型的最终能力。因此预训练数据要具备较好的多样性，覆盖不同的表达方式及知识范围，这样才可提高模型的泛化能力。

* 质量高：模型参数量大，学习能力强。如果存在低质数据，模型也会从中充分学习，进而会影响模型的表达方式、知识的准确性以及安全性，导致一系列难以解决的问题。因此数据质量十分重要，其不仅要具备正确的表达方式和准确的知识，同时内容也要尽可能的安全。此外，预训练数据要尽量避免重复。

为构建一份合理的预训练数据，我们从不同来源收集了大量且丰富的训练预料。同时对数据进行了不同的处理与加工来保证数据质量。在下面内容中，将详细介绍我们预训练数据的构成、量级和数据加工方法。

#### 数据构成

为保证数据的量级及多样性，我们构建的预训练数据包含中、英文，涉及7个不同的领域。经过滤和去重后，数据类型及对应的token数量如下：

**中文：**
* 内容分享类、对话问答——0.53T
* 书籍教材类——0.06T
* web数据（common crawl为主）——0.93T
* 知识百科类——0.02T
* 新闻类——0.052T
* 论文、期刊——0.04T
* 金融类——0.1T

**英文：**
* 内容分享类、对话问答——0.021T
* 书籍教材类——0.026T
* Web 数据（CulturaX为主）——3T
* 知识百科类——0.024T

数据的分布如下图所示：

<img src=resources/pretraining_data.png width=70%>

#### 数据处理

为保证预训练数据的质量，我们采取了一系列方法对数据进行处理，包括规则清洗、PPL过滤、数据去重和内容安全过滤。下面将介绍数据处理的具体内容。

**规则清洗：** 规则清洗主要按照预先设计的一系列规则来对数据进行规范化和过滤。规则清洗速度较快，但只能清洗特定模式的数据，无法深入理解文本内容。尽管如此，规则清洗也可解决大量有明显问题的数据。具体而言，规则清洗主要包括以下几方面：

* 格式规范化：主要包括中文繁简转换，标点符号全半角统一、重复符号压缩处理；
* 关键词过滤（篇章级）：使用恶意关键词过滤明显有害的文本；
* 长度过滤（行级别）：过滤掉过短或无效片段。

**PPL过滤：** 除了衡量大语言模型的文本建模能力外，PPL[<sup>1</sup>](#ppl)也是一个很好的衡量数据内容质量的指标。我们使用统计语言模型针对PPL过高的文本进行过滤，保留语义流畅的高质量文本内容。

**数据去重：** 我们使用MinHash[<sup>2</sup>](#minhash)方案进行重复数据过滤，主要包括如下步骤：MinHash生成、构建LSH索引和相似的pair、根据相似的pair求连通图、全局去重。

**内容安全过滤：** 为更大程度缓解大模型的安全隐患，我们需要进一步对预训练数据进行安全过滤。目标是过滤掉垃圾广告、政治敏感、暴力、色情等不符合人类价值观的数据。为了保持模型的泛化性，不能直接将以上几个领域的数据全部直接删除。因此我们详细制定了不同领域的内容标准，人工标注有害和无害样本，训练多个内容有害分类器。再经过人工评估，确定分类器的阈值，在清洗掉有害文本的前提下，避免大量误伤文本。

在预训练中，我们并没有使用全部的数据，而是按照一定的策略从每个领域中采样一定量的数据进行模型训练。

### 训练

与其他语言模型（LLMs）类似，我们采用了类似于LLaMA[<sup>3</sup>](#llama)<sup>,</sup>[<sup>4</sup>](#llama2)框架的结构。我们的模型具有4096个隐藏单元，由30层和32个注意力头组成。为了融入位置信息，我们采用了RoPE[<sup>5</sup>](#rope)作为位置嵌入技术。模型中使用的激活函数是SwiGLU[<sup>6</sup>](#swiglu)，并使用RMSNorm[<sup>7</sup>](#rmsnorm)进行归一化处理。在训练过程中，我们将最大序列长度设置为2048个token。词表的大小为39438，与我们先前模型（XuanYuan-13B、XuanYuan-70B）使用的词表一致。这些设置和架构选择使得我们的模型能够有效地捕捉和处理给定数据中的复杂语言模式和依赖关系。为了高效训练我们的模型，我们使用NVIDIA A800 80GB GPU以及DeepSpeed[<sup>8</sup>](#deepspeed)分布式训练框架，在DeepSpeed中，我们使用zero[<sup>9</sup>](#zero) stage 1。下表展示了我们模型的具体配置。

| models       | size | hidden size | layers | heads | position embedding | activation function | normalization | vocab   | length |
| ------------ | ---- | ----------- | ------ | ----- | ------------------ | ------------------- | ------------- | ------- | ------ |
| LLaMA        | 7B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 32,000  | 2,048  |
| LLaMA2       | 7B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 32,000  | 4,096  |
| ChatGLM-6B   | 6B   | 4096        | 28     | 32    | ROPE               | GELU                | Layer Norm    | 130,528 | 2,048  |
| ChatGLM2-6B  | 6B   | 4096        | 28     | 32    | ROPE               | SwiGLU              | RMSNorm       | 65,024  | 32,768 |
| Baichuan-7B  | 7B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 64,000  | 4,096  |
| Baichuan2-7B | 7B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 125,696 | 4,096  |
| Qwen-7B      | 7B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 151,851 | 2,048  |
| Yi-6B        | 6B   | 4096        | 32     | 32    | ROPE               | SwiGLU              | RMSNorm       | 64,000  | 4,096  |
|XuanYuan-6B   | 6B   | 4096        | 30     | 32    | ROPE               | SwiGLU              | RMSNorm       | 39,438  | 2,048  |

在训练过程中，我们使用了各种类型的数据，包括新闻文章、用户生成内容、研究论文、书籍和代码示例等。这种多样的数据来源丰富了训练语料库，使模型能够从不同的领域和文本类型中学习。为了确保模型的性能提升，我们在训练过程中采用动态评估和调整的方法。具体而言，我们在每个检查点对模型在特定任务或基准上的性能进行评估，并根据评估结果动态调整不同来源的训练数据配比。通过不断监控模型的训练进展并微调数据分布，我们可以不断优化模型训练过程，提升模型的各项能力。

为了增强模型的考试能力，我们采用了额外的考试题和知识点。这确保了模型在考试场景中能够提供准确的解答。为了实现这一点，我们利用离线搜索增强技术，根据给定的上下文检索相应的考试题和答案。这种方法使得模型能够利用更广泛的与考试相关的信息，并提高其生成精确答案的能力。此外，为了使模型能够以生成式的方式获取知识，我们将选择题的提示和选项转化为相应的知识点。这种方法使得模型能够从具体实例中进行泛化，并在考试和问答环境中灵活应用其所掌握的知识。通过采用这种方法，模型对底层概念有了更深入的理解，并能够提供更全面和有见地的回答。

下表展示了预训练的超参数配置：

| Hyperparameter      | XuanYuan-6B |
| ------------------- | ----------- |
| Batch size / device | 6           |
| Learning rate       | 5e-4        |
| Min. learning rate  | 1e-6        |
| Warmup steps        | 50          |
| Gradient clipping   | 1.0         |

## 有监督微调

### 数据构造

在有监督微调中，数据的选择是成功的基石。数据的质量、数量和配比都扮演着至关重要的角色。质量决定了模型学习的内容，数量影响着模型对任务的适应性，而配比则平衡了模型对通用知识与任务特定知识的利用。以下是具体的几个方面：
* 数据质量：选择高质量的数据至关重要，因为它确保了模型学习到的内容是准确、一致且具有代表性的。标注的正确性、内容的准确性和数据的多样性都是关键因素。不良的数据质量可能导致模型学习到错误的模式，降低其泛化能力，甚至引发过拟合。
* 数据数量：数据量的选择也至关重要。充足的数据有助于模型更好地学习任务的特征和模式，提高其性能。然而，过多的数据也可能增加计算资源的负担，导致训练时间延长，并可能引入噪声。因此，需要在数据的有效性和训练成本之间找到平衡。
* 数据配比：在微调过程中，需要将预训练数据与特定任务数据合理配比。这需要根据任务需求进行调整，以确保模型既能学习到通用语言知识，又能掌握任务特定知识。在某些情况下，增加特定任务数据的比例可能有助于提高模型在该任务上的性能；而在其他情况下，保持一定的通用知识可能更为关键。
* 数据多样性与代表性：为了确保模型在真实场景中具有良好的泛化能力，所选择的数据应覆盖特定任务的各种场景、语境和类型。同时，选择来自不同来源和类型的数据可以提高模型的鲁棒性。

为了实现这些目标，我们采用self-QA[<sup>10</sup>](#selfqa)方法进行指令微调数据的收集。首先，我们简要介绍问题生成（QG）和问题回答（QA）这两个密切相关的任务。它们可以被视为一个对偶问题，前者涉及根据给定的段落或信息生成问题，后者涉及根据给定的段落或信息回答问题。特别是，机器阅读理解（MRC）技术通常用于QA任务。对于人类来说，自问自答学习意味着根据提供的信息提出问题并回答问题，然后将其回答与原始知识进行比较。这种方法在增强个体对提供信息的理解方面非常有效。对于特定领域的指令样本，我们可以将指令和输入视为一个整体，因此指令和输入可以等同于问题，而指令输出则等同于回答。

Self-QA方法能够在没有人工标注的情况下生成大量高质量的问答数据，为模型提供有监督的训练样本。基本思想是利用现有的高质量大型模型，根据无监督的知识生成微调数据。这些无监督的知识可以来自书籍、网页，也可以是从结构化的表格数据或图谱数据转换而来的非结构化文档数据。具体流程如下：

<img src=resources/6b_self_qa.png width=60%>

1. 知识引导的指令生成：使用语言模型ChatGPT根据无监督文本生成领域相关的指令。为了确保指令不依赖于参考文本内容，需要提供一些准则。这样就可以获得多个相关的指令，在下一个阶段使用。无监督的知识数据可以是连续文本，如书籍或网页，也可以是经过预处理的非结构化文本数据，如表格或知识图谱。对于这些结构化数据将采用下面的方法转换成非结构化数据：

<img src=resources/6b_stru_cha.png width=45%>

在这里使用的prompt如下：

<img src=resources/6b_ins_gen.png width=60%>

2. 机器阅读理解：在这个阶段，语言模型根据无监督的知识对生成的指令问题进行阅读理解，并生成答案。这里使用的prompt如下：

<img src=resources/6b_read_compre_prom.png width=60%>

3. 修剪与过滤：尽管在前面的阶段中规定了一些生成问题和答案的限制，但仍然会有一些违反规则的文本生成。此外，生成的指令示例可能也存在格式问题。因此，需要应用不同的启发式过滤器来确保生成的文本符合预定义的准则，并保持正确性和连贯性。经过过滤后的问题和答案可以直接用作指令微调数据。此外，还可以将指令和相应的无监督知识添加到问题中，以便模型学习特定的领域任务，如阅读理解和信息提取。通过以上步骤，self-QA方法可以在没有指令数据的情况下，利用高质量的大型模型从无监督文档中生成指令数据，并通过有监督微调提高模型在遵循指令方面的能力。

### 模型训练

在模型训练层面，我们使用混合微调训练的方式进行。虽然通用大型模型的使用越来越广泛，但特定领域模型的重要性也不可忽视。在许多领域中，语言的分布和特定的语言细微差别需要进行针对性的微调或专门训练的模型。因此，已经出现了一系列特定领域的大型模型，以满足不同领域的需求。特定领域的语言模型和聊天模型对数据分发和训练方法提出了更高的要求。它们需要捕捉特定领域的语言特征、术语和上下文，以实现最佳性能。然而，仅仅依靠特定领域的数据进行训练可能会导致灾难性的遗忘，即模型失去了之前从通用领域学到的知识，从而影响整体性能。混合微调训练框架的基本思想是结合通用领域和特定领域的数据进行训练，以获得更好的性能和更广泛的适应性。

混合微调方法是一种有效的策略，在微调阶段中巧妙地结合了无监督预训练数据和有监督指令微调数据，以避免灾难性遗忘的发生。无监督预训练数据可以通过从互联网抓取并进行清理和过滤来获取。至于有监督指令微调数据，我们采用了自我指导和自我问答等方法进行收集。混合微调方法的优势在于，它充分利用了预训练模型在大规模无监督数据上所学到的语言表示能力，并通过有监督指令微调数据提供任务特定的指导。通过混合无监督数据和有监督数据，包括通用和特定领域的数据，模型能够在微调过程中保持对预训练知识的记忆，从而避免灾难性遗忘的问题。这种方法不仅可以提高模型在特定任务上的性能，还能够增强其泛化能力和适应性。

## 强化对齐

基于人类反馈的强化学习（Reinforcement learning with human feedback，RLHF）是对大语言模型（Large language model，LLM）进行对齐的有效手段。参考Instruct-GPT[<sup>11</sup>](#instruct_gpt)和LLaMA2[<sup>4</sup>](#llama2)中的做法，我们也对指令微调后的XuanYuan-6B进行了RLHF训练，以进一步对齐模型表现与人类偏好，提高模型通用性、安全性及金融能力。

具体而言，RLHF过程一般包括三个步骤：偏好数据构建、奖励模型（Reward model，RM）训练及强化训练。接下来我们分别介绍每个步骤的具体做法。

### 偏好数据构建

偏好数据中包含了人类的偏好信息，一条偏好数据一般由4方面构成，即：

<span id="data_form"></span> $$(x, y_1, y_2, l) \tag{1}$$

其中 $x$ 为prompt， $y_1$ 和 $y_2$ 为prompt $x$ 的两条response， $l$ 为偏好标注信息，其标注了 $y_1$ 和 $y_2$ 哪个更符合人类偏好（在给定 $x$ 的条件下）。由此可见，要构建偏好数据集，我们要有prompt、prompt 对应的两个（或多个）response，同时还需要对数据进行偏好标注。

#### Prompt构建

在构建prompt时，我们重点关注两方面，一方面是数据的丰富度与多样性，一方面是数据的质量。

为保证数据的多样性，我们把通用性、安全性及金融属性进行了更细粒度的拆分，得到了多个子项，并按照一定的量级和比例收集每一子项的数据。这样可以使收集的prompt覆盖到不同的方面，同时具备合理的量级及配比。通用性的拆分以及各子项数据配比见下图。

<img src=resources/prompt_parts.png width=70%>

为保证数据质量，我们聘请了专业人员对数据进行了清洗：删除或修改有明显错误的prompt、改进表达或格式有瑕疵的prompt、保留正确的prompt。经过清洗后，我们获得了4w+高质量的prompt数据。

我们相信，数据至关重要。如果数据配比不合适，或者有重大质量问题，无论采用的算法多高明，学习到的模型也必然是不合理的。

#### Response生成

为保证RM训练数据和测试数据分布的一致性，避免出现OOD（Out of distribution）问题，我们使用XuanYuan-6B-SFT来产生response。这是因为RM测试数据为actor模型的输出（强化训练阶段），而actor的初始状态为XuanYuan-6B-SFT。

使用XuanYuan-6B-SFT的采样参数，在生成多个response时，response间彼此相似度较高，难以标注偏好信息。因此，我们提高了采样参数中temperature和top_p的值，然后再生成response，以保证response的多样性，以及其包含的偏好信息的多样性。

#### 偏好标注

当前业界流行的偏好标注方式主要有两种：rank标注及pair标注。rank标注中，一个prompt包含多个response（一般为4个），标注者要求对多个response进行排序。之后根据排序信息，可以将response两两组合，构建形如公式(1)所示的偏好数据。Instruct-GPT[<sup>11</sup>](#instruct_gpt)即采用这类标注方式。相比rank标注，pair标注则更为直接，一条prompt仅生成两个response，标注者直接比较两个response，标出哪条response更符合偏好。此外，一些标注方法也要求标出偏好的强度。Anthropic[<sup>12</sup>](#anthropic)和LLaMA2[<sup>4</sup>](#llama2)即采用pair形式的偏好标注。

初期，我们采用rank标注方式，但在实际操作时，发现这种标注方式一方面标注速度较慢，另一方面不同的标注人员标注结果的一致性较低。为了解决这个问题，我们也采用了pair标注方式，同时也要求标注出偏好的强度，以收集更多的偏好信息，来提升RM的泛化性能。下图展示了我们具体的标注页面，标注结果有8个档位，从左到右依次命名为A3、A2、A1、A0、B0、B1、B2、B3。其中A3表示A好于B的程度最高，B3表示B好于A的程度最高，其他档位依此类推。

<img src=resources/label_platform.png>

数据标注质量十分重要，如果标注的数据本身存在问题，那么模型训练绝对不会正常。为了提升标注质量，我们制定了一套完善的标注标准，覆盖了实际中可能出现的大多数场景，并在标注过程中不断发现和解决新出现的问题，不断扩充完善我们的标注标准。此外，我们对标注人员进行了深入的培训和指导，让他们能真正理解和使用这套标准，并在他们标注中遇到疑惑时及时解答。最后，我们对交付的标注结果进行了严格的质检，如数据不合格会重新进行标注，直至满足验收标准。通过一系列的优化措施，我们获得了一批覆盖面广、质量高的偏好数据。这些认真构建的数据是我们得以成功进行RLHF训练的关键。

### RM训练

#### 架构

我们使用XuanYuan-6B-SFT作为RM的基本架构。偏好数据中的response由XuanYuan-6B-SFT生成，因此XuanYuan-6B-SFT对偏好数据有更好的适配性，可以快速理解偏好数据并进行偏好建模。

对于XuanYuan-6B-SFT，我们去掉最后的LM_head layer，并将其替换为value head layer。Value head layer为一个线性层，输入是的XuanYuan-6B-SFT次顶层的特征，输出为一个一维的reward分数。训练开始时，我们将value head layer进行随机初始化。训练过程中，value head layer和底层XuanYuan-6B-SFT（已去掉LM_head layer）联合进行训练。

#### 数据

我们使用构建的偏好数据进行RM训练，但删除了偏好强度最低的数据（即A0，B0）。偏好强度低意味着两个response比较接近，未包含明显的偏好信息。这类数据歧义比较大，会让模型感觉比较“困惑”，不利于模型进行偏好建模。

在实际操作中，我们共使用了约6w+偏好数据，其中90%用于训练，剩余的10%用于测试。

#### 损失函数

我们使用对比损失进行RM训练，根据Bradley-Terry (BT)模型[<sup>13</sup>](#bt_model)，偏好分布可写成如下形式：

$$p(y_1 \succ y_2 \mid x) = \sigma[r_\theta(x,y_1) - r_\theta(x,y_2)] \tag{2}$$

其中 $x$ 为prompt， $y_1$ 和 $y_2$ 为 $x$ 的两个response， $\sigma$ 为sigmoid函数， $r(x,y)$ 为prompt $x$ 和相的response $y$ 的reward， $\theta$ 为RM的参数。给定公式(2)，偏好数据集 $\mathcal{D} = \lbrace x^{(i)}, y_c^{(i)}, y_r^{(i)}\rbrace_{i=1}^{N}$ 的负log似然（均值）为：

$$-\mathbb{E}\_{(x,y_c,y_r) \in \mathcal{D}}[\log \sigma(r_\theta(x,y_c) - r_\theta(x, y_r))] \tag{3}$$

其中 $y_c$ 为chosen resposne， $y_r$ 为rejected response，公式3即为训练RM常用的对比损失。

参考DeepSpeed-Chat[<sup>14</sup>](#dsc)中做法，我们使用token-level的对比损失来进行RM的训练。对于 $y_c$ 和 $y_r$，先找到他们第一个不相同的token所在的位置，作为起始位置；然后找到两个response结束的位置，并取两者的最大值，作为结束位置；之后计算从起始位置，到结束位置，相同位置上 $y_c$ 和 $y_r$ 之间的对比损失，最后求对比损失的均值作为该条偏好样本的损失。在预测阶段，我们取response最后一个token对应的reward作为该response的reward。

为保证训练/测试的一致性，训练时应该取 $y_c$ 最后一个token的reward和 $y_r$ 最后一个token的reward来计算对比损失，我们称这种对比损失为sentence-level损失。我们实验对比了两种损失函数的表现，结果表明sentence-level损失训练RM可获得更高的测试精度。但是RM不仅用于打reward分，还用于强化训练阶段critic model的初始化。我们发现使用sentence-level损失训练的RM初始化critic model后，强化训练会变得不稳定，难以收敛。因此我们仍使用token-level损失来进行RM训练，虽然精度会有小幅下降，但强化训练的稳定性会有较大提高。

#### 模型选择

在RM训练阶段，我们会训练多个epoch，并在每个epoch结束后存储当前RM，之后选择合适的RM进行后续强化训练。在选择RM时，我们主要看以下几点：

* 测试精度：因为测试精度客观反映了RM打分合理性；

* RM输出的reward尺度：如果reward值过小或过大，在后续强化训练时会产生数值问题，导致训练无法正常进行；

* chosen和rejected response reward之间的margin：具体做法是计算测试集中的chosen response reward均值和rejected response reward均值，观察两个均值之间是否存在一定的margin。如存在一定的margin，则说明RM有较强的鲁棒性。

我们使用的RM测试精度是63%，输出尺度在[-1, 1]区间内，margin为0.5。

### 强化训练

#### 结构

强化训练中，actor model和reference model为XuanYuan-6B-SFT，critic model和reward model均采用XuanYuan-6B-SFT作为基本架构，critic model用训练好的RM进行初始化。训练中，actor model和critic model需要进行更新，而reference model和reward model则保持不变。

#### 数据

强化训练的数据为prompt数据，即

$$\mathcal{D} = \lbrace x^{(i)}\rbrace_{i=1}^{M} \tag{4}$$

偏好数据本身也包含prompt，其中的prompt也可以用来进行强化训练。RM是基于偏好数据训练的，对于偏好数据中的prompt及对应的response，RM对其打分会更为精准（相较其他prompt而言）。所以将偏好数据中的prompt用于强化训练会使训练过程更为“容易”，很大程度上可以避免RM打分不准而导致的一系列问题，如reward hacking、训练不收敛等。

但是，仅采用偏好数据中的prompt是不够的，这样模型见到的数据过于局限，不利于提升模型的泛化性能。因此我们增加了额外的新prompt一起用于强化训练。新prompt的构建方式（配比、清洗方法等）和偏好数据中[prompt构建](#prompt构建)方式相同。此外，偏好数据prompt和新增prompt的比例为1:1。

#### 训练

在进行强化训练时，我们参考了Instruct-GPT[<sup>11</sup>](#instruct_gpt)、LLaMA2[<sup>4</sup>](#llama2)以及Anthropic[<sup>12</sup>](#anthropic)的做法。在实现上，我们参考了DeepSpeed-Chat[<sup>14</sup>](#dsc)框架。

强化训练的目标是优化actor，在给定prompt $x$ 的条件下，actor生成的response $y$ 能获得更高的期望回报 $R$，即：

$$\arg\max_{\pi}\mathbb{E}_{x\in D, y \in \pi}[R(y|x)] \tag{5}$$

其中 $\pi$ 表示待优化的策略，也就是actor model。回报 $R$ 由两部分组成，一部分是RM对生成的response的评分 $r_\theta(x,y)$；另一部分是生成 $y$ 时，actor $\pi$ 和初始模型（即XuanYuan-6B-SFT） $\pi_0$ 之间的KL散度，这主要是约束actor，不让其偏离SFT模型太远。这一方面可一定程度上减轻强化训练的“对齐税”，另一方面也有利于PPO训练。具体而言，回报 $R$ 可被记作：

$$R(y|x)=r_\theta(y|x)-\beta KL[\pi(y|x) \lVert \pi_0(y|x)] \tag{6}$$

我们使用Proximal policy optimization (PPO)[<sup>15</sup>](#ppo)方法来优化上述目标函数。具体而言，actor每生成一个token被视作一次强化决策，其对应的奖励是在进行token预测时，actor model $\pi$ 和初始模型 $\pi_0$ 的LM-Head输出的概率分布间的KL散度。当生成完整个response后，我们将其输入RM获得reward，并将RM的reward叠加到最后一个时刻的奖励值上。基于奖励值，我们使用GAE[<sup>16</sup>](#gae)方法计算优势函数，并进一步计算出PPO loss和critic loss，然后进行误差反传计算梯度，并用梯度更新actor model和critic model。

生成经验时，我们使用了和SFT模型相同的采样参数，一方面使模型有一定的探索空间，另一方面也保证了生成response的质量。我们设置KL的权重 $\beta=0.05$；过高的 $\beta$ 会使模型接近初始模型 $\pi_0$，强化训练效果不明显；过低的 $\beta$ 会过度优化reward值，容易造成reward hacking。我们将actor model和critic model的学习率设置为5e-7；过高的学习率会让RM值快速上升，容易造成reward hacking；而过低的学习率会极大降低训练速度。在计算loss时，我们使用fp32的数据精度，避免loss的数值问题引起的训练不稳定现象。我们共训练了约300 PPO step，训练中我们重点关注critic loss和RM reward值的变化，critic loss整体上应呈现下降趋势，而RM reward整体上应呈现上升趋势，但RM reward升的过高也是一种异常现象，此时大概率出现了reward hacking。

#### 模型选择

每训练20个PPO step，我们会存储当前的actor model。训练完成后，根据RM reward变化情况，我们挑选几个不同阶段的代表性模型进行快速的人工评估。人工评估时对比对象是强化训练前的模型，即XuanYuan-6B-SFT。评估完成后统计good（actor response $\succ$ SFT model response），same（actor response = SFT model response），bad（actor response $\prec$ SFT model resposne）数量，选择最有优势的actor model进行更正式的人工评估。

### 模型评估

我们聘请了专业的评估人员进行模型评估，评估题目覆盖通用性、安全性、金融垂类等不同范畴。为避免不同评估人员的喜好偏差，每道题目均由三个不同的评估人员进行评估。评估题目对其他人员完全封闭，避免研发人员通过构造类似的评估题目进行训练来获得更好的评估结果。在评估时，我们的对比对象是XuanYuan-6B-SFT，希望经过强化训练后能进一步提升SFT模型的效果。

下图展示了模型在通用性（评估时安全性被纳入了通用性范畴）和金融能力的综合评估结果。从图中可以看出，在两个领域，经过RLHF训练后，模型的能力都有了极大的提升。这证明了我们RLHF训练的有效性。

<img src=resources/6b_all_eval.png width=60%>

下图展示了模型在通用性的各细分子领域评估结果。从结果来看，在大多数子领域，经过强化训练后，模型的能力都有了明显的提高。在日常对话、逻辑推理、内容创作和安全性等子领域，强化带来的效果提升尤为明显。这些结果再次证明了强化对齐的有效性。然而在一些其他子领域，比如信息摘要、翻译等，强化训练并未带来明显的进步。在后续工作中，我们会补充更多的偏好数据，同时提升偏好标注质量，来进一步补齐这些弱项的能力。

<img src=resources/6b_general_eval.png width=60%>

下图展示了模型在金融各细分子领域的评估结果。从图中看出，在金融知识理解、金融业务分析两个子领域，强化训练带来了明显的能力提升。而在其他子领域，强化训练并未取得预期的效果。对这些子领域，我们同样会补充更多高质量的偏好数据，提高RM对这类prompt和response打分的准确性，进而提升强化训练的效果。

<img src=resources/6b_finance_eval.png width=60%>


## 参考文献
<div id="ppl"></div>

1. Jelinek, Fred, et al. "Perplexity—a measure of the difficulty of speech recognition tasks." *The Journal of the Acoustical Society of America* 62.S1 (1977): S63-S63.

<div id="minhash"></div>

2. Broder, Andrei Z. "On the resemblance and containment of documents." *Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171)*. IEEE, 1997.

<div id="llama"></div>

3. Touvron H, Lavril T, Izacard G, et al. Llama: Open and efficient foundation language models[J]. arXiv preprint _arXiv:2302.13971_, 2023.

<div id="llama2"></div>

4. Touvron, Hugo, et al. “Llama 2: Open foundation and fine-tuned chat models.” _arXiv preprint arXiv:2307.09288_ (2023).

<div id="rope"></div>

5. Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." *Neurocomputing* 568 (2024): 127063.

<div id="swiglu"></div>

6. Shazeer, Noam. "Glu variants improve transformer." *arXiv preprint arXiv:2002.05202* (2020).

<div id="rmsnorm"></div>

7. Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." *Advances in Neural Information Processing Systems* 32 (2019).

<div id="deepspeed"></div>

8. Rasley, Jeff, et al. "Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters." *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2020.

<div id="zero"></div>

9. Rajbhandari, Samyam, et al. "Zero: Memory optimizations toward training trillion parameter models." *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2020.

<div id="selfqa"></div>

10. Zhang, Xuanyu, and Qing Yang. "Self-QA: Unsupervised Knowledge Guided Language Model Alignment." *arXiv preprint arXiv:2305.11952* (2023).

<div id="instruct_gpt"></div>

11. Ouyang, Long, et al. “Training language models to follow instructions with human feedback.” _Advances in Neural Information Processing Systems_ 35 (2022): 27730-27744.

<div id="anthropic"></div>

12. Bai, Yuntao, et al. “Training a helpful and harmless assistant with reinforcement learning from human feedback.” _arXiv preprint arXiv:2204.05862_ (2022).

<div id="bt_model"></div>

13. Bradley R A, Terry M E. Rank analysis of incomplete block designs: I. The method of paired comparisons[J]. _Biometrika_, 1952, 39(3/4): 324-345.

<div id="dsc"></div>

14. Yao, Zhewei, et al. "Deepspeed-chat: Easy, fast and affordable rlhf training of chatgpt-like models at all scales." _arXiv preprint arXiv:2308.01320_ (2023).

<div id="ppo"></div>

15. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347,_ 2017.

<div id="gae"></div>

16. Jaques, Natasha, et al. "Way off-policy batch deep reinforcement learning of implicit human preferences in dialog." _arXiv preprint arXiv:1907.00456_ (2019).


# 轩辕：首个千亿级中文金融对话模型
XuanYuan: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters

## 1. 模型简介
轩辕是国内首个开源的千亿级中文对话大模型，同时也是首个针对中文金融领域优化的千亿级开源对话大模型。轩辕在BLOOM-176B的基础上针对中文通用领域和金融领域进行了针对性的预训练与微调，它不仅可以应对通用领域的问题，也可以解答与金融相关的各类问题，为用户提供准确、全面的金融信息和建议。

## 2. 模型下载
千亿级的轩辕模型已可以在[Huggingface](https://huggingface.co/xyz-nlp/XuanYuan2.0)中申请下载。

## 3. 模型评测
### 3.1. 金融领域效果评测
为了深入评估轩辕对话大模型在金融领域的专业性，我们构建了一套针对金融领域的详细评测体系。该评测体系涉及金融名词理解、金融市场评论、金融数据分析和金融新闻理解等四大领域，从多角度全面考察模型的金融处理能力。评测结果表明，轩辕在主流的四种开源大模型的比较中，赢得了150次回答中63.33%的胜率，凸显出其在金融领域的显著优势。

<img width="457" alt="f119d2cf76493fc383aef1e45" src="https://github.com/Duxiaoman-DI/XuanYuan/assets/10808269/18ae1d5f-c9be-4813-8386-db6dc5f4896e">


<details>
<summary>点击显示轩辕回答示例</summary>

| 问题 | 答案 |
| --- | --- |
|请解释一下"股票回购"是什么意思，以及公司为什么要进行股票回购?|股票回购（Share buyback）是指公司回购其已发行的股票，以注销或留作未来使用。回购股票可以减少公司的股本，从而提高每股股息和每股收益。<br><br>公司进行股票回购的原因有很多，以下是一些常见的原因：<br><br>1. 提高每股股息和每股收益：通过回购股票，公司可以减少股本，从而提高每股股息和每股收益。<br><br>2. 提高股价：回购股票可以减少公司的股本，从而提高每股股息和每股收益，进而提高股价。<br><br>3. 减少负债：通过回购股票，公司可以减少负债，从而减少利息支出。<br><br>4. 提高股东回报率：通过回购股票，公司可以减少股本，从而提高每股股息和每股收益，进而提高股东回报率。<br><br>5. 改善公司形象：通过回购股票，公司可以减少股本，从而提高每股股息和每股收益，进而改善公司形象。<br><br>需要注意的是，股票回购可能会对公司的财务状况产生影响，因此公司需要谨慎考虑是否进行股票回购。|
|你能解释一下什么是"授信额度"吗？|授信额度是指银行或金融机构向客户提供的最高贷款额度，即客户可以在此额度内自由支配的借贷金额。通常情况下，银行会根据客户的个人信用记录、收入情况以及负债情况等因素来决定授信额度的大小。授信额度是银行和金融机构向客户提供信贷服务的重要指标之一。<br>授信额度的具体计算方式会因不同的银行和金融机构而异，但通常包括以下几个方面：<br>1.个人信用记录：银行和金融机构通常会考虑客户的信用记录，包括借款还款记录、信用卡使用情况等，这些因素将直接影响到客户获得授信额度的概率。<br>2.收入水平：银行和金融机构通常会要求客户提供稳定的收入来源，例如工资收入、投资回报等。银行和金融机构会通过审核客户的收入证明和财务报告来确定客户的授信额度。<br>3.负债情况：银行和金融机构通常会关注客户的负债情况，包括其他未偿还贷款、信用卡欠款等。如果客户已经存在高额的负债负担，那么银行和金融机构可能会对其授信额度进行限制。<br>4.担保能力：如果客户没有良好的信用记录或者收入状况较差，银行和金融机构可能会要求客户提供担保，例如房产、车辆等财产作为担保，以提高授信额度的可能性。<br>总之，授信额度是一种重要的借贷工具，可以帮助客户在资金需求时更加灵活地应对各种场合。|
|你能解释一下什么是"抵押贷款"和"信用贷款"的区别吗？|抵押贷款和信用贷款都是用于借款的方式，但它们之间的主要区别在于贷款金额、利率、还款期限以及担保方式。<br><br>抵押贷款是指将财产（如房屋、车辆等）作为担保来获取贷款。在抵押贷款中，贷款金额通常是按照财产的价值计算的，因此贷款金额较大。此外，由于有抵押物作为担保，银行或金融机构通常会提供更低的利率。还款期限通常较长，因为贷款人需要足够的时间来还清贷款并归还抵押物。一旦借款人无法按时还款，银行或金融机构有权处置抵押物以收回欠款。<br><br>信用贷款则是指根据个人信用记录、收入和支出情况等因素来决定是否给予贷款。在信用贷款中，银行或金融机构会基于个人的信用评分来决定贷款额度和利率。由于没有抵押品作为担保，因此贷款金额相对较小。还款期限较短，通常在6个月到3年之间，还款期数也较为灵活。如果借款人无法按时还款，银行或金融机构有权停止发放贷款。<br><br>总之，抵押贷款和信用贷款的区别在于贷款金额、利率、还款期限和担保方式。选择哪种贷款方式取决于个人的财务状况和需求。|
|你能解释一下什么是"银行业务的KYC"流程吗？|KYC是"Know Your Customer"的缩写，意思是了解您的客户。在银行业务中，KYC流程是指在客户开立银行账户之前，银行需要对客户进行身份验证和背景调查的过程。KYC流程的目的是确保客户的身份和财务状况真实可靠，并防止洗钱和恐怖主义融资等非法活动。<br><br>KYC流程通常包括以下步骤：<br><br>1. 客户身份验证：银行会要求客户提供身份证明文件，如身份证、护照等，并核实客户的身份信息是否真实有效。<br><br>2. 客户背景调查：银行会对客户进行背景调查，了解客户的职业、收入来源、家庭状况等信息，以确认客户的财务状况是否真实可靠。<br><br>3. 客户风险评估：银行会对客户进行风险评估，以确定客户的风险等级，并采取相应的风险控制措施。<br><br>4. 客户反洗钱合规审查：银行会对客户进行反洗钱合规审查，以确保客户的交易行为合法合规，并防止洗钱和恐怖主义融资等非法活动。<br><br>总之，KYC流程是银行业务中非常重要的一环，它确保了客户的身份和财务状况真实可靠，并防止了洗钱和恐怖主义融资等非法活动。|
|"牛市"和"熊市"分别代表什么意思？它们对投资者有什么样的影响？|股票市场中的牛市和熊市是指股票价格的涨跌情况。牛市通常指股票价格持续上涨的情况，而熊市则指股票价格持续下跌的情况。<br><br>对于投资者来说，牛市和熊市对他们的影响是不同的。在牛市中，投资者通常会感到乐观和兴奋，因为股票价格上涨，他们可以赚取更多的利润。在熊市中，投资者通常会感到悲观和沮丧，因为股票价格下跌，他们可能会损失更多的资金。<br><br>在牛市中，投资者通常会采取积极的投资策略，例如购买更多的股票，以期获得更高的回报。在熊市中，投资者通常会采取保守的投资策略，例如减少股票投资，以降低风险。<br><br>需要注意的是，牛市和熊市只是股票市场中的一种情况，投资者应该根据自己的投资目标和风险承受能力来选择合适的投资策略。在牛市中，投资者可以采取高风险高回报的投资策略，而在熊市中，投资者应该采取低风险低回报的投资策略。此外，投资者还应该密切关注市场动态，及时调整自己的投资策略，以适应市场的变化。|
|欧洲央行（ECB）表示，将继续实施宽松的货币政策，以支持欧元区经济复苏。然而，通货膨胀压力和全球经济形势的不确定性给ECB的政策制定带来了挑战。请分析欧洲央行维持宽松货币政策的原因，以及这一政策可能对欧元区经济和全球金融市场产生哪些影响？|维持宽松货币政策的主要原因是欧洲央行希望通过降低利率和增加货币供应量来刺激经济增长和创造就业机会。此外，ECB还希望通过降低利率和增加货币供应量来控制通货膨胀压力。<br><br>欧洲央行的宽松货币政策可能对欧元区经济和全球金融市场产生以下影响：<br><br>1. 欧元区经济增长：ECB的宽松货币政策可能会刺激欧元区的经济增长，因为低利率和增加货币供应量可以促进投资和消费。<br><br>2. 通货膨胀压力：ECB的宽松货币政策可能会导致通货膨胀压力增加，因为货币供应量增加可能导致物价上涨。<br><br>3. 全球金融市场：ECB的宽松货币政策可能会对全球金融市场产生影响，因为欧洲央行的货币政策是全球货币政策的重要组成部分。ECB的宽松货币政策可能会导致美元贬值，从而对全球金融市场产生影响。<br><br>总之，ECB的宽松货币政策可能会对欧元区经济和全球金融市场产生积极影响，但也需要注意通货膨胀压力和全球经济形势的不确定性所带来的挑战。|
|全球各大央行纷纷推出量化宽松政策，以应对新冠疫情对经济的影响。然而，有观点认为，这些政策可能导致资产泡沫和通胀压力上升。请分析量化宽松政策可能导致的资产泡沫和通胀风险，以及政府和央行如何平衡经济复苏与通胀风险之间的关系？|量化宽松政策是指央行通过购买国债等金融资产来增加货币供应量，以刺激经济增长。然而，这些政策可能导致资产泡沫和通胀压力上升。<br><br>首先，量化宽松政策可能导致资产泡沫。当央行购买国债等金融资产时，这些资产的价格可能会上涨，从而形成资产泡沫。如果资产泡沫破裂，可能会导致经济衰退和金融危机。<br><br>其次，量化宽松政策可能导致通胀压力上升。当央行购买国债等金融资产时，这些资产的价格可能会上涨，从而导致物价上涨。如果通胀压力过高，可能会对经济造成负面影响。<br><br>政府和央行如何平衡经济复苏与通胀风险之间的关系？<br><br>政府和央行可以通过以下措施来平衡经济复苏与通胀风险之间的关系：<br><br>1. 控制货币供应量。央行可以通过调整货币政策来控制货币供应量，从而控制通胀压力。<br><br>2. 监管金融市场。政府可以通过监管金融市场来防止资产泡沫的形成。<br><br>3. 调整利率政策。央行可以通过调整利率政策来控制通胀压力，从而平衡经济复苏与通胀风险之间的关系。|
|如果一家公司的股票价格是每股100美元，而其每股派发的股息是5美元，那么其股息收益率是多少？|股息收益率（dividend yield）是指公司每股派发的股息与股票价格之间的比率。计算股息收益率的公式为：股息收益率 = 每股派发的股息 ÷ 股票价格。<br><br>根据题目，这家公司的股票价格是每股100美元，每股派发的股息是5美元。因此，这家公司的股息收益率为：<br><br>股息收益率 = 每股派发的股息 ÷ 股票价格 = 5美元 ÷ 100美元 = 0.05 = 5%<br><br>因此，这家公司的股息收益率为5%。这意味着，如果你持有这家公司的股票，你每年可以获得5美元的股息收益。<br><br>需要注意的是，股息收益率只考虑了每股派发的股息和股票价格之间的关系，而没有考虑其他因素，如公司的盈利能力、市场前景等。因此，股息收益率只是一个参考指标，不能完全代表一家公司的投资价值。在做出投资决策时，还需要考虑其他因素，如公司的财务状况、行业前景等。|
|如果一家公司的总资产为500亿美元，总负债为300亿美元，股东权益是多少？|	股东权益可以通过总资产减去总负债来计算，即：<br><br>股东权益 = 总资产 - 总负债<br><br>根据题目中给出的数据，总资产为500亿美元，总负债为300亿美元，因此：<br><br>股东权益 = 500亿美元 - 300亿美元 = 200亿美元<br><br>因此，这家公司的股东权益为200亿美元。<br><br>需要注意的是，股东权益并不等同于净资产。净资产是指公司的总资产减去总负债后的余额，即：<br><br>净资产 = 总资产 - 总负债<br><br>根据题目中给出的数据，总资产为500亿美元，总负债为300亿美元，因此：<br><br>净资产 = 500亿美元 - 300亿美元 = 200亿美元<br><br>因此，这家公司的净资产为200亿美元。<br><br>需要注意的是，净资产并不等同于股东权益。股东权益是指公司归属于股东的所有权益，包括股本、资本公积、盈余公积、未分配利润等。而净资产是指公司的总资产减去总负债后的余额，包括股本、资本公积、盈余公积、未分配利润等。因此，股东权益和净资产是两个不同的概念。|
|如果一家公司的总资产为200亿美元，总负债为100亿美元，那么其资产负债率是多少？|	资产负债率（debt-to-equity ratio）是指公司负债总额与股东权益总额的比率，通常用于衡量公司的财务风险和偿债能力。计算资产负债率的公式为：<br><br>资产负债率 = 总负债 / 总资产<br><br>根据题目给出的信息，这家公司的总资产为200亿美元，总负债为100亿美元，因此其资产负债率为：<br><br>资产负债率 = 100亿美元 / 200亿美元 = 0.5 = 50%<br><br>因此，这家公司的资产负债率为50%。资产负债率越高，表示公司负债越多，财务风险越大，偿债能力越弱。反之，资产负债率越低，表示公司负债越少，财务风险越小，偿债能力越强。<br><br>需要注意的是，资产负债率并不是唯一的衡量公司财务风险和偿债能力的指标。其他指标，如流动比率、速动比率、利息保障倍数等，也可以用来评估公司的偿债能力和财务风险。同时，不同行业和不同类型的公司，其资产负债率和财务指标也会有所不同。因此，在评估公司的财务状况时，需要综合考虑多个指标和因素。|
|请解释一下什么是股息收益率，它如何帮助我们评估一家公司的股息政策？|	股息收益率（dividend yield）是指公司每年派发的股息金额与股票市价的比率，通常以百分比的形式表示。股息收益率可以帮助我们评估一家公司的股息政策。<br><br>股息收益率越高，意味着公司每年派发的股息金额越多，相对于股票市价的比率也越高。因此，股息收益率越高，意味着投资者可以获得更高的股息收益。<br><br>股息收益率可以帮助我们评估一家公司的股息政策。如果一家公司的股息收益率相对较高，那么这可能意味着该公司更注重股息派发，并希望通过股息来回报投资者。相反，如果一家公司的股息收益率相对较低，那么这可能意味着该公司更注重股票价格的上涨，并希望通过股票价格的上涨来回报投资者。<br><br>需要注意的是，股息收益率只是评估一家公司股息政策的一个指标。投资者还需要考虑其他因素，例如公司的财务状况、行业前景、管理层能力等。因此，在评估一家公司的股息政策时，需要综合考虑多个因素，而不仅仅是股息收益率。|

</details>

### 3.2. 通用领域效果评测
除了在金融领域的突出表现，轩辕还在通用处理能力方面表现优秀。我们利用由200道多元化问题构成的通用评测集，对轩辕在数学计算、场景写作、逻辑推理、文本摘要等13个主要维度的表现进行了评估，并将其与ChatGPT进行了对比。结果显示，轩辕在这个评测集上，有71%的问题表现不亚于ChatGPT，验证了其全方面能力。

<img width="481" alt="9038f05c34b82b3eae00603f6" src="https://github.com/Duxiaoman-DI/XuanYuan/assets/10808269/f85989c5-7e68-4fee-b9dc-39ef1c500c06">

## 4. 相关论文

如果有用到轩辕相关方法和模型，请引用以下论文：

[XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters ](https://arxiv.org/abs/2305.12002)

[Self-QA: Unsupervised Knowledge Guided Language Model Alignment ](https://arxiv.org/abs/2305.11952)

[CGCE: A Chinese Generative Chat Evaluation Benchmark for General and Financial Domains](https://arxiv.org/abs/2305.14471)

## 5.使用说明
本模型推荐运行在8卡A100 GPU或同等配置下以获得最佳性能。

可以通过以下代码调用本模型：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME", trust_remote_code=True)
model = AutoModel.from_pretrained("MODEL_NAME", trust_remote_code=True)
```

【热门问题】如何调用轩辕模型？

由于本模型较大并不支持线上API测试，请下载模型后使用transformers库的AutoTokenizer和AutoModel进行调用。
轩辕对话模型的输入示例：

```text
BOS_TOKEN + "Human: " + query + "\n\nAssistant: "
```

轩辕对话模型的生成示例：

```python
output = model.generate(**input, do_sample=True, temperature=0.8, top_k=50, top_p=0.9, early_stopping=True, repetition_penalty=1.1, min_new_tokens=1, max_new_tokens=256)
```

## 6. 免责声明与许可协议
轩辕作为一个开源的中文金融对话模型，仅限于非商业用途的目的。该模型的设计初衷是为了促进学术研究、技术探索和个人学习等非商业领域的应用。我们鼓励学术界、开发者和研究人员使用轩辕来推动对话系统和金融领域的进步。其中，商业用途包括但不限于将轩辕用于产品、服务、咨询等与商业利益相关的活动。

对于轩辕模型生成的言论，我们不承担任何责任。使用者在将轩辕应用于非商业用途时，需要自行承担潜在的风险，并始终保持审慎。我们建议用户在使用模型输出的信息时，进行独立的验证和判断，并根据个人的需求和情境进行决策。我们希望通过轩辕的开源发布，为学术界和开发者社区提供一个有益的工具，并推动对话系统和金融技术的发展。我们鼓励大家积极探索和创新，以进一步拓展和应用轩辕的潜力，并共同促进人工智能在金融领域的研究和实践。

## 7. 总结
我们鼓励使用者在相关工作中引用轩辕，以促进知识的交流和分享，并推动中文金融对话系统的发展。轩辕的发布将为金融领域的应用和研究提供强大的支持，并为中文金融对话系统的发展做出重要贡献。我们期待看到更多的创新和应用，以提升金融服务和用户体验，并进一步推动人工智能技术在金融领域的发展。

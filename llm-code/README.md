# Open LLM Flow
### 运行环境
Nvidia显卡驱动版本>=525.85.12  
CUDA版本>=12.3  
Python版本>=3.10.12  
gcc版本>=11.4.0  
  
python pip安装包版本如下：  
argparse==1.4.0  
deepspeed==0.12.5  
datasets==2.15.0  
transformers==4.36.0  
sentencepiece==0.1.99  


### 数据集下载
方式一：直接下载，数据集文件可通过如下huggingface链接下载  
https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus  

方式二：python代码获取，代码如下
```py
from datasets import load_dataset

dataset = load_dataset("Duxiaoman-DI/FinCorpus")
```

### 模型下载
方式一：直接下载，Llama-2-7b-hf模型文件可通过如下huggingface链接下载  
https://huggingface.co/meta-llama/Llama-2-7b-hf  
  
方式二：python代码获取，代码如下
```py
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### 执行步骤
```sh
切换到你的宿主机工作目录
cd /your_host_workspace

clone项目代码
git clone open-llm-flow

切换到容器内项目根目录
cd open-llm-flow

删除缓存
sh clear_cache.sh

执行数据预处理
sh data_preprocess_run.sh

执行预训练
sh pretrain_run.sh

执行sft
sh sft_run.sh
```

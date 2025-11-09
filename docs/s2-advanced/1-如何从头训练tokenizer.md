# 一、导入训练tokenizer所需库

`tokenizers`是Hugging Face出的一个高性能、可定制的子词分词器库，主要用于训练和使用像BPE、WordPiece、Unigram等子词模型，是训练LLM时常用的工具。

```python
import random
import json
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)
import os
```

`Tokenizer`是核心分词器对象，控制整个分词、编码、解码过程，可以与不同的模型、预处理器和解码器配合使用。

`models`包含各种子词分词模型（如BPE、WordPiece、Unigram），定义了如何对文本进行分割与映射成token IDs。

`pre_tokenizers`定义了文本的预处理方式，负责在真正的分词前对文本进行初步分割，如按空格、字节或其他规则分割。

`trainers`用于训练分词模型的工具，包括设置词表大小、特殊符号等的参数配置，常用的有`BpeTrainer`、`WordPieceTrainer`等。

`decoders`用于将分词后的token IDs转回原始文本（解码），支持不同的解码策略，如`ByteLevel`和`Metaspace`。

# 二、初始化tokenizer

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

代码首先定义了一个使用BPE模型的分词器，该分词器使用BPE算法将文本拆分成子词单元，以增强模型对未登录词和低频词的处理能力。

然后将预处理器设置为ByteLevel，这一步将文本转换为字节级别的单位，允许更细粒度的文本处理，且add_prefix_space=False控制是否在每个单词前加空格，由于处理的是中文，因此将其设置为False。

# 三、设置训练器trainer并添加特殊token

```python
# 定义特殊token
special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# 设置训练器并添加特殊token
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,  # 确保这三个token被包含
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
```

这些特殊tokens是在训练过程中专门定义的，用于标识特定的文本模式或结构，通常用于控制生成文本的结构。

* "<|endoftext|>"用于表示文本的结束，通常在模型生成任务中用来指示生成文本的终止。
* "<|im_start|>"标识对话或任务的开始，可以用于标记输入文本的起始。
* "<|im_end|>"标识对话或任务的结束，用来标记输入文本的结束。

这些特殊tokens会在训练过程中作为词表的一部分，确保它们在分词和生成过程中能被正确处理。

trainer的参数解释：

* `vocab_size=6400`表示模型训练过程中会生成最多6400个子词（包括特殊tokens）。
* `special_tokens=special_tokens`保证特殊tokens在BPE训练过程中不会被拆分或合并。
* `show_progress=True`在训练过程中显示进度。
* `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`指定BPE初始字母表为ByteLevel默认字母表，以适应不同字符和符号类型。

# 四、读取文本数据

使用预训练数据集训练tokenizer，为了便于演示，这里只读取前100条数据。

```python
# 读取JSONL文件并提取文本数据
def read_texts_from_jsonl(file_path, max_samples=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            yield data['text']

data_path = r'D:\MyFile\github\minimind-master\minimind_dataset\pretrain_hq.jsonl'
texts = read_texts_from_jsonl(data_path)
```

查看数据示例：

```python
print(list(texts)[1])
```

```
<|im_start|>根据输入的内容，编写一个类别标签。
这是一篇介绍如何阅读心电图的文章类别标签: 医学/心电图阅读指南<|im_end|> <|im_start|>帮我搜索一下最近的天气情况。当然，我可以帮您搜索最新的天气情况。请问您需要查询哪个城市的天气情况呢？<|im_end|> <|im_start|>帮我讲一个令人开心的笑话。好的，我帮您讲一个关于细菌的笑话。为什么细菌不会上网？因为连接总是断开了！<|im_end|> ...
```

# 五、开始训练tokenizer

```python
# 训练tokenizer
tokenizer.train_from_iterator(texts, trainer=trainer)
```

# 六、设置解码器

为分词器设置ByteLevel解码器，让其在将token ID序列转换回原始文本时，能够正确还原按字节切分的内容。

```python
tokenizer.decoder = decoders.ByteLevel()
```

# 七、保存训练好的tokenizer

```python
tokenizer_dir = r"./model"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)
```

# 八、手动创建并保存配置文件

```python
# 手动创建配置文件
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "added_tokens_decoder": {
        "0": {"content":"<|endoftext|>","lstrip":False,"normalized":False,"rstrip":False,"single_word":False,"special":True},
        "1": {"content":"<|im_start|>","lstrip":False,"normalized":False,"rstrip":False,"single_word":False,"special":True},
        "2": {"content":"<|im_end|>","lstrip":False,"normalized":False,"rstrip":False,"single_word":False,"special":True}
    },
    "additional_special_tokens": [],
    "bos_token": "<|im_start|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "legacy": True,
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<|endoftext|>",
    "chat_template": "{% if messages[0]['role']=='system' %}{% set system_message=messages[0]['content'] %}{{'<|im_start|>system\\n'+system_message+'<|im_end|>\\n'}}{% else %}{{'<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n'}}{% endif %}{% for message in messages %}{% set content=message['content'] %}{% if message['role']=='user' %}{{'<|im_start|>user\\n'+content+'<|im_end|>\\n<|im_start|>assistant\\n'}}{% elif message['role']=='assistant' %}{{content+'<|im_end|>'+'\\n'}}{% endif %}{% endfor %}"
}

# 保存配置文件
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)
```

| 字段名 | 解释 |
|--------|------|
| `add_bos_token` | 是否自动在文本开头添加 `bos_token`（如 `<|im_start|>`）。False表示不添加。|
| `add_eos_token` | 是否自动在文本末尾添加 `eos_token`（如 `<|im_end|>`）。False表示不添加。|
| `add_prefix_space` | Byte-level 分词时是否在文本前加空格。中文设为False。|
| `added_tokens_decoder` | 特殊token的详细配置，包括内容、是否为特殊token等。key为内部token ID。|
| `additional_special_tokens` | 除了 `bos/eos/pad/unk` 外，额外声明的特殊token列表。当前为空。|
| `bos_token` | 起始token，通常用于语言模型开头控制符。|
| `clean_up_tokenization_spaces` | 解码时是否清理token化带来的空格冗余。False表示不清理。|
| `eos_token` | 结束token，通常用于语言模型输出结束标记。|
| `legacy` | 设置为True兼容旧版本tokenizer行为。|
| `model_max_length` | 模型支持的最大token长度。超过将触发截断或报错。|
| `pad_token` | 用于对齐padding的特殊token。|
| `sp_model_kwargs` | SentencePiece模型的额外配置参数，BPE未使用为空。|
| `spaces_between_special_tokens` | 是否在特殊token之间自动添加空格。False。|
| `tokenizer_class` | 指定tokenizer类型。Hugging Face使用"PreTrainedTokenizerFast"加速。|
| `unk_token` | 用于标记未知词的token。|
| `chat_template` | Jinja2模板字符串，用于格式化对话数据为模型输入格式。|

# 九、测试训练好的tokenizer

```python
def eval_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(r"D:\MyFile\github\minimind-master\mm")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)

    print('\n输入文本：\n',new_prompt,'\n')   
    print('解码文本：\n',response,'\n')  

eval_tokenizer()
```

```
tokenizer实际词表长度： 259
encoder长度： 133
decoder和原始文本是否一致： True

输入文本：
 <|im_start|>system
你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
<|im_start|>user
你来自哪里？<|im_end|>
<|im_start|>assistant
我来自地球<|im_end|>

解码文本：
 <|im_start|>system
你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
<|im_start|>user
你来自哪里？<|im_end|>
<|im_start|>assistant
我来自地球<|im_end|>
```

至此，关于`minimind`中的tokenizer训练部分解读完成。

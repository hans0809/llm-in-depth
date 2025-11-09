
在上一篇文章中，我们通过对预训练的GPT-2进行微调，得到了一个垃圾邮件分类器。事实上，这种方式是使用GPT-2的网络作为backbone，在其输出后接一个分类头，来完成二分类任务。

在本文中，我们将介绍另一种微调方式：指令微调（Instruction Tuning）。

通过指令微调，我们可以打造一个对话机器人，就像你一直在使用的各种大语言模型应用那样 —— 它能够接收用户的自然语言指令，并输出相应的回复。


# 一、什么是指令微调？
指令微调（Instruction Tuning） 是一种让预训练语言模型学会“听懂人话”的方法。它的核心思想是：通过监督微调（Supervised Fine-Tuning, SFT），让模型学习从「指令（Instruction）」到「输出（Response）」的映射。

这种方式与传统的分类、回归等任务不同，指令微调的数据格式通常是自然语言对话格式：

```
用户：请告诉我Python中如何定义一个函数？
助手：你可以使用`def`关键词，例如：
def my_function():
    print("Hello World")
```

在训练阶段，我们通常提供大量这样的指令-响应对，模型在学习之后就能够泛化到未见过的指令上进行合理回答。

# 二、指令微调的数据格式(附数据集下载链接)
指令微调的数据格式通常是一个Instruction-Response（指令-回复）对。

典型的数据格式如下（以JSON为例）：
```
{
  "instruction": "请简要介绍一下Python语言。",
  "input": "",
  "output": "Python是一种高级编程语言，具有简洁的语法和丰富的库，广泛应用于数据科学、Web 开发、自动化等领域。"
}
```

或者，有时也可以带上input字段作为补充输入信息，比如：
```
{
  "instruction": "请根据以下文本总结重点内容。",
  "input": "Python 是一种广泛使用的编程语言，具有丰富的生态系统。",
  "output": "Python 拥有广泛的应用和丰富的生态系统，是一门流行的编程语言。"
}
```

本文使用的数据集链接为：https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json

里面包含了1100个符合指令微调要求的数据样本对。示例如下：
![](../images/s1-foundations/LLM指令微调：训练一个人工智能助手-大模型炼丹术(八)/1.png)

将这些数据存储到data变量，并划分到训练集、验证集和测试集：
```python
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
```
查看：
```python
print("Example entry:\n", train_data[5])
```
输出：
```
Example entry:
 {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}
```

# 三、如何转成模型输入格式？

为了适配GPT类模型的自回归训练方式，我们需要将输入和输出拼接在一起作为一个序列，并让模型学习在给定指令后生成合适的响应。

这里，在构造模型的input-target pair时，仍然遵循我们之前做预训练时所介绍的向右偏移一位的方法。

也就是说，**指令微调并不改变GPT模型的结构本质，而是通过构造“指令+输入+输出”的文本串，让模型在语言建模中学会完成任务式的生成。标签就是这个串整体向右平移一位。**


比如，以上面介绍的第二个样本数据为例，将其转换为模型输入的格式，如下：
```
### instruction：
请根据以下文本总结重点内容。

### input:
Python 是一种广泛使用的编程语言，具有丰富的生态系统。

### output:
Python 拥有广泛的应用和丰富的生态系统，是一门流行的编程语言。

```
这一整个是一个文本序列，也是模型的输入数据格式。


# 四、构建指令微调的数据加载器

定义一个数据加载器，将上面1100个样本对转换到模型的输入格式，即：将每一条数据中的 instruction、input 和 output 拼接成一个完整的文本序列。
```
Below is an instruction that describes a task.  
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```
代码如下：
```python
import torch
from torch.utils.data import Dataset
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# 自定义一个 Dataset 类，便于与 DataLoader 配合使用
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 原始的 instruction 数据（ list of dict）

        # 对文本进行预编码，减少训练时重复工作
        self.encoded_texts = []
        for entry in data:
            # 构造 Prompt 部分（instruction + input）
            instruction_plus_input = format_input(entry)

            # 构造 Response 部分（即模型需要学习生成的内容）
            response_text = f"\n\n### Response:\n{entry['output']}"

            # 拼接完整输入：Prompt + Response
            full_text = instruction_plus_input + response_text

            # 使用 tokenizer 将文本编码为 token id 序列
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 返回编码后的 token 序列（暂时未进行 padding）
        return self.encoded_texts[index]

    def __len__(self):
        # 数据集中样本数量
        return len(self.data)

# 一个辅助函数，用于构建 Prompt 部分（instruction + optional input）
def format_input(entry):
    # Instruction 文本（每条样本必须包含）
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # Input 文本（某些任务可能为空）
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # 返回完整的 Prompt 部分（Instruction + Input）
    return instruction_text + input_text

```

PyTorch的DataLoader会从Dataset（这里是InstructionDataset）中拿到多个样本，然后会调用collate_fn(batch)把这些样本打包成一个batch。默认行为是用default_collate（就是stack）。但对于像GPT这种 输入长度不同、还需要做input-target 对构造的任务，必须自己写一个collate_fn。

custom_collate_fn是在使用DataLoader加载训练数据时，控制“如何把多个样本拼成一个batch”的关键函数。它直接决定了送入模型的输入和目标长什么样。

```python
def custom_collate_fn(
    batch,
    pad_token_id=50256,         # 用于 padding 的 token，一般是 <|endoftext|>（GPT 的 pad）
    ignore_index=-100,          # 用于目标（target）中 mask 掉 padding 区域，避免影响 loss
    allowed_max_length=None,    # 可选，限制序列的最大长度，节约资源
    device="cpu"                
):
    # 获取当前 batch 中的最大长度（每个样本都添加了一个 endoftext token）
    batch_max_length = max(len(item)+1 for item in batch)

    # 初始化用于保存最终输入与目标的列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 在序列末尾添加一个 endoftext token（GPT 的格式）
        new_item += [pad_token_id]

        # 将序列 pad 到 batch 中最大长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        # 构造输入：去掉最后一个 token（GPT 是用 input 预测下一个 token）
        inputs = torch.tensor(padded[:-1])
        # 构造目标：去掉第一个 token，相当于 input 的右移一位
        targets = torch.tensor(padded[1:])

        # 替换 target 中除第一个 padding 外的所有 padding 为 ignore_index（不参与 loss 计算）
        mask = targets == pad_token_id                # 找到 pad 位置
        indices = torch.nonzero(mask).squeeze()       # 获取 pad 的索引
        if indices.numel() > 1:                       # 如果有多个 padding
            targets[indices[1:]] = ignore_index       # 忽略除第一个之外的所有 padding 区域

        # 如果设置了最大序列长度，进一步裁剪 input 和 target
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # 添加到 batch 列表
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将所有样本拼接成一个 batch，并移动到指定设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
```

举个例子，假设 InstructionDataset 里有两条样本：
```python
batch = [
    [1, 2, 3],         # 第一条数据，token_ids
    [4, 5]             # 第二条数据，短一点
]
```

设置：
```python
pad_token_id = 50256
ignore_index = -100
```

以下是将batch使用custom_collate_fn进行转换的中间变化过程：
```
1. 加 <|endoftext|> 后：

[1, 2, 3, 50256]
[4, 5, 50256]

2. 补齐（最长是 4）：

[1, 2, 3, 50256]        # 已满，无需 padding
[4, 5, 50256, 50256]    # 多加一个 pad

3. 构造 input 和 target（分别左移 & 右移）：

inputs:  [1, 2, 3, 50256]       target: [2, 3, 50256, 50256]
inputs:  [4, 5, 50256, 50256]   target: [5, 50256, 50256, 50256]

4. 把 target 中除了第一个 pad 的地方都替换为 -100（ignore_index）：

target: [2, 3, 50256, -100]
target: [5, 50256, -100, -100]

5. 最后组成两个张量传给模型训练：
inputs_tensor.shape  => [batch_size=2, seq_len=4]
targets_tensor.shape => [batch_size=2, seq_len=4]
```

ok，现在数据加载器所需的组件已经准备好了，把他们组合在一起，构建数据加载器：
```python
from functools import partial
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
```

实例化查看数据加载器的输出：
```python
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
```

输出：
```
Train loader:
torch.Size([8, 61]) torch.Size([8, 61])
torch.Size([8, 76]) torch.Size([8, 76])
torch.Size([8, 73]) torch.Size([8, 73])
torch.Size([8, 68]) torch.Size([8, 68])
torch.Size([8, 65]) torch.Size([8, 65])
torch.Size([8, 72]) torch.Size([8, 72])
torch.Size([8, 80]) torch.Size([8, 80])
torch.Size([8, 67]) torch.Size([8, 67])
torch.Size([8, 62]) torch.Size([8, 62])
torch.Size([8, 75]) torch.Size([8, 75])
...
```
可以看到，每一个batch的最大序列长度会有所不同，这个最大长度就是当前batch中最长序列的长度，其它较短的序列会被padding以匹配这个最大长度。

这种方式能够确保每个batch的处理是动态的，避免了固定max_length可能带来的浪费（如果固定长度过长，很多短序列会浪费计算资源；如果固定长度过短，则会导致信息丢失）。同时，通过动态计算batch内最长序列长度，而不是统一使用一个较大的固定长度，可以更高效地利用内存。

# 五、LLM的指令微调
这一部分和之前的预训练代码几乎一致，这里直接搬过来：
```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

开始指令微调，这里作为演示只训练一个epoch：
```python
import time

start_time = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 1

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

可视化loss曲线：
```python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```
![](../images/s1-foundations/LLM指令微调：训练一个人工智能助手-大模型炼丹术(八)/2.png)

可以看到，随着训练的进行，loss正常下降。

# 六、推理
现在，使用经过指令微调的模型进行一些推理测试：

```python
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")



with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
```

输出：
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Rewrite the sentence using a simile.

### Input:
The car is very fast.

Correct response:
>> The car is as fast as lightning.

Model response:
>> The car is as fast as a bullet.
-------------------------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What type of cloud is typically associated with thunderstorms?

Correct response:
>> The type of cloud typically associated with thunderstorms is cumulonimbus.

Model response:
>> A thunderstorm is a type of cloud that typically forms in the atmosphere over a region of high pressure. It typically produces a strong wind that blows across the area, creating a dense, dense cloud.
-------------------------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Name the author of 'Pride and Prejudice'.

Correct response:
>> Jane Austen.

Model response:
>> The author of 'Pride and Prejudice' is George Bernard Shaw.
-------------------------------------
```

至此，恭喜你你已经学习了基础篇的全部知识内容，完结撒花~~

准备好，进阶篇马上开始！
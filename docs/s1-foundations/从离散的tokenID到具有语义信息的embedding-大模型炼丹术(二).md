
使用tokenizer将文本转换成token序列的过程，被称为“tokenization”。在完成了tokenization之后，我们已经可以将一个个的单词映射到对应的数字，称之为token ID，这些数字已经可以被计算机处理。然而，若直接将这些数字应用于模型训练，仍存在一些问题：

1. 缺乏语义信息：
    token ID只是一个索引，本身不包含任何语义信息。例如，“cat” 可能被映射为ID 1254，而“dog”是ID 3920，这两个ID之间的数值关系是无意义的。直接使用它们可能会导致模型误解token之间的关联性。

2. 整数之间的数值关系会误导模型：
    机器学习模型通常会学习数据之间的模式。如果直接输入token ID，模型可能会误以为ID 1254（"cat"）和ID 3920（"dog"）之间存在某种数学关系（如加减乘除），但实际上ID只是索引，没有数值上的逻辑关系。

3. 无法捕捉相似token的关系
    语义相近的token在embedding空间中应该具有接近的表示。例如，"king"和"queen"应该在高维空间中比较接近，而"apple"和"computer"应该相距较远。然而，单纯的token ID无法提供这种分布信息。

带着这些问题，往下看~

# 一、什么是Token Embedding ?
Token Embedding（标记嵌入）是将离散的token（如单词、子词或字符）转换为连续的向量表示的过程。在自然语言处理任务中，神经网络无法直接处理文本，需要将文本转换为有语义特征的数字形式，而Token Embedding就是这一转换的核心步骤。

在PyTorch中，使用`torch.nn.Embedding`来构建embedding层，实际上，这是一个查找表，key是token id，value是token id对应的embedding向量。

举个例子，假设词汇表大小是6，embedding维度是3，那么，构建的embedding layer的weight的shape是`6x3`：
```python
# 构建embedding layer
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)# 6x3
```
输出：
```
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```
这个就是embedding层的weight，也就是一个查找表，它有6行3列，6是词汇表大小，3是embedding层的维度。

假设输入句子转换到token id后如下：
```python
input_ids = torch.tensor([2, 3, 5, 1])
```
其中，2，3，5，1其实就是要在当前查找表中查询的索引，因此
```python
print(embedding_layer(input_ids))
```
的输出如下：
```
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
```

到这里，我们已经可以将原始的输入文本转换成embedding向量，具体包括两个过程：

- 1）先通过tokenization将文本拆分为token ID序列

- 2）然后通过 Token Embedding 层 获取 token 的连续表示，即embedding

# 二、如何训练Embedding层 ?

Embedding层本质上是一个可学习的查找表，其核心是一个嵌入矩阵（Embedding Matrix），用于将离散的token ID映射到连续的高维向量。这个矩阵的大小（shape）通常取决于词表大小和嵌入维度，例如：

```python
vocab_size = 5000    # 词表大小
embedding_dim = 256  # 嵌入维度
embedding_matrix = torch.nn.Embedding(vocab_size, embedding_dim)
print(embedding_matrix.weight.shape)  # 输出: torch.Size([5000, 256])
```

在训练过程中（例如文本分类任务，有文本的类别标签），模型会自动更新Embedding矩阵的权重，使得语义相似的token向量彼此接近，而语义不相关的token向量距离较远。

通常，在构建自己的任务时，会加载预训练的embedding，这样可以充分利用已有大规模语料训练得到的语义信息，提高模型性能，并加速收敛。


# 三、nn.Embedding 和 nn.Linear的区别是什么 ?

Embedding相当于一个固定输入为one-hot向量的Linear层。

假设`num_embeddings=10, embedding_dim=4`，那么Embedding其实是一个形状为`[10, 4]`的权重矩阵E。

如果用Linear模拟Embedding：

1. 先把token ID转换成one-hot向量（形状 `[10]`）。
2. 然后进行矩阵乘法（one-hot只会选中对应的行，相当于输入的X是1，与weight matrix行相乘的结果就是weight matrix行）。

代码如下：
```python
# 创建一个相当于 nn.Embedding 的 Linear 层
embedding_as_linear = nn.Linear(in_features=10, out_features=4, bias=False)

# 生成一个 one-hot 形式的输入（假设词汇表大小 10）
one_hot_input = torch.eye(10)[[1, 3, 5]]  # 选中索引 1、3、5
print(one_hot_input.shape)  # (3, 10)

# 进行计算
word_vectors = embedding_as_linear(one_hot_input)
print(word_vectors.shape)  # (3, 4)
```

其中：

```
embedding_as_linear.weight.T:
Parameter containing:
tensor([[ 0.2629, -0.0249, -0.0409, -0.1259],
        [-0.0418,  0.0100, -0.2939,  0.0127],
        [-0.2674, -0.0493, -0.1956, -0.0738],
        [-0.0907,  0.0497,  0.2699, -0.0348],
        [-0.2227,  0.2818,  0.0189, -0.3083],
        [ 0.0209,  0.1934, -0.2562,  0.1481],
        [-0.0590,  0.1122,  0.0499,  0.2776],
        [-0.1696,  0.0687,  0.2613,  0.1933],
        [-0.0288,  0.0746, -0.2988, -0.2239],
        [ 0.2996,  0.1222, -0.2129, -0.2549]], grad_fn=<PermuteBackward0>)


one_hot_input:
tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])


word_vectors:
tensor([[-0.0418,  0.0100, -0.2939,  0.0127],
        [-0.0907,  0.0497,  0.2699, -0.0348],
        [ 0.0209,  0.1934, -0.2562,  0.1481]], grad_fn=<MmBackward0>)
```


# 四、为什么还需要Positional Embedding？

位置编码（Positional Encoding, PE）是一种用于在无序输入（如Transformer）中引入位置信息的技术，其编码得到的向量称为Positional Embedding。由于Transformer不使用循环（RNN）或卷积（CNN），它无法捕捉序列顺序，因此需要额外的信息来表示单词的顺序。

Transformer采用自注意力机制（Self-Attention），它本质上是对输入进行加权求和，不考虑输入的顺序。例如：

```
句子1：“小猫在沙发上睡觉。”
句子2：“沙发在小猫上睡觉。”
```

在LSTM/RNN中，单词的顺序会通过循环网络的隐状态传递体现出来，而在Transformer中，所有单词是同时处理的，所以：

* 没有位置编码：模型会认为两句话的含义是一样的！
* 有位置编码：模型能够识别单词的顺序，从而理解句子的语义。

# 五、常见的位置编码方式

## 1. 按照参数是否可以学习来分类

按照参数是否是可学习的，可以分为：可学习的（Learnable）位置编码，固定的（Sinusoidal）位置编码。

### 可学习的（Learnable）位置编码

直接为每个位置训练一个可学习的向量，类似于nn.Embedding，让模型自动学习最佳的位置信息。

```python
# 假设最大序列长度为100，每个位置用512维向量表示
max_len = 100
d_model = 512  # 词向量维度

# 可学习的位置编码
positional_embedding = nn.Embedding(max_len, d_model)

# 生成10个token的位置信息
positions = torch.arange(10).unsqueeze(0)  # shape: (1, 10)
pe = positional_embedding(positions)  # shape: (1, 10, 512)
```

* 优点：训练时可以自动调整位置编码的权重；适用于特定任务，如果训练数据的序列长度固定，可以使用这种方法。  
* 缺点：不能推广到比训练时更长的序列。

### 固定的（Sinusoidal）位置编码

Transformer论文提出了一种固定的位置编码方法，利用正弦（sin）和余弦（cos）函数，使不同位置的编码具有唯一性，并且可以推广到更长的序列，其代码实现如下：

```python
import numpy as np
import torch

def positional_encoding(seq_len, d_model):
    """
    生成 Transformer 位置编码
    :param seq_len: 序列长度
    :param d_model: 词向量维度
    :return: 位置编码矩阵 (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度使用sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度使用cos

    return torch.tensor(pe, dtype=torch.float32)

# 生成10个token的位置编码，词向量维度512
pe = positional_encoding(10, 512)
print(pe.shape)  # torch.Size([10, 512])
```

* 优点：无需训练，可直接计算；可以推广到更长的序列，不会因为训练长度限制而失效。  
* 缺点：对特定任务可能没有可学习的位置编码效果好。

## 2. 按照位置关系分类

在位置编码的讨论中，“绝对”和“相对”是根据编码方式的依赖关系来区分的。它们的核心区别在于位置编码所表示的内容：是单独表示每个token在序列中的具体位置（绝对），还是表示token之间的相对位置关系（相对）。

### 绝对位置编码

绝对位置编码为每个token分配一个唯一的、固定的标识符，这个标识符表示该token在序列中的位置，其特点如下：

* 位置编码是根据每个token在序列中的位置生成的，不考虑token之间的相对距离或顺序。  
* 每个位置都有一个固定的编码值，即每个位置的编码与其绝对位置直接相关。

上面介绍的根据正弦（sin）和余弦（cos）函数设计的固定位置编码就是一种绝对位置编码。

### 相对位置编码

相对位置编码表示token之间的相对位置，即某个token与其他token之间的相对位置关系，而不是每个token的绝对位置，其特点如下：

* 位置编码关注的是token之间的相对关系，而不依赖于它们的绝对位置。这样就能将模型从单一的固定序列长度中解耦，提供更大的灵活性。  
* 相对位置编码通常不为每个位置定义一个唯一的标识符，而是为每对token之间的相对距离生成一个编码（如相对位置的偏置或因子）。

# 六、GPT使用哪种位置编码方式 ？
**GPT系列使用的是三角函数的绝对位置编码。**

GPT系列模型是基于自回归的Transformer架构，目的是生成序列（如文本）。使用绝对位置编码有助于模型理解token在序列中的顺序，以便在生成文本时考虑上下文信息。

相较于相对位置编码，绝对位置编码较为简单，适用于很多任务，尤其是在模型输入序列长度相对固定或有限的情况下。由于GPT系列通常处理的输入长度相对较短，绝对位置编码是足够有效的。

此外，Llama系列模型和一些其他的模型采用了 旋转位置编码（RoPE, Rotary Position Embedding）。这种位置编码方式不同于传统的绝对位置编码和相对位置编码，它通过旋转的方式来处理序列中的位置信息。具体的实现细节会在进阶篇中详细介绍。

ok，现在已经将句子映射到了embedding，那么embedding在后续将会被如何处理呢？它在神经网络中的数据流向又是怎样的呢？所有内容将在下篇文章中继续介绍，欢迎持续关注。

参考：
- [1] https://0809zheng.github.io/2022/07/01/posencode.html
- [2] https://www.qin.news/jue-dui-wei-zhi-bian-ma-he-xiang-dui-wei-zhi-bian-ma/

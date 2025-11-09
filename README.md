# 项目名称

一个带你 **快速入门大语言模型（LLM）** 的系列教程项目，从 LLM 基础原理到实践操作，循序渐进讲解核心概念与技能。

本项目分为两部分：  
- **基础篇《大模型炼丹术》**：面向具有Python与深度学习基础的LLM初学者，讲解 LLM 核心概念与基础实现；  
- **进阶篇《MiniMind 源码解读》**：面向进阶读者，在学习完基础篇后，进一步深入 LLM 内部机制，讲解现代化大语言模型的关键技术与优化方法。


## 项目受众

* 有Python与深度学习基础，想快速掌握LLM（只看基础篇即可）
* 想了解现代LLM架构与训练技巧（基础篇+进阶篇），并动手实践的学生和开发者

## 目录
### 基础篇--《大模型炼丹术》 1️⃣
> 💡 基础篇主要介绍核心概念，帮助你快速上手 LLM。

学完基础篇，你将掌握 LLM 的核心知识，包括：

- Tokenizer 与数据预处理  
- Embedding 原理  
- Causal Attention  
- 从零搭建 GPT-2 架构  
- 自回归预训练  
- 微调与指令微调  

| 序号 | 文章链接 |
|------|----------|
| 1 | [大模型炼丹术(一)：从 tokenizer 开始，为自回归预训练准备数据集](./docs/s1-foundations/从tokenizer说起，为LLM自回归预训练准备数据集-大模型炼丹术(一).md) |
| 2 | [大模型炼丹术(二)：从离散 token IDs 到具有语义信息的 embedding](./docs/s1-foundations/从离散的tokenID到具有语义信息的embedding-大模型炼丹术(二).md) |
| 3 | [大模型炼丹术(三)：从单头到多头，深度解析注意力机制](./docs/s1-foundations/从单头到多头，深度解析大模型的注意力机制-大模型炼丹术(三).md) |
| 4 | [大模型炼丹术(四)：动手搭建 GPT-2 架构](./docs/s1-foundations/动手搭建GPT2架构-大模型炼丹术(四).md) |
| 5 | [大模型炼丹术(五)：LLM 自回归预训练过程详解](./docs/s1-foundations/LLM自回归预训练过程详解-大模型炼丹术(五).md) |
| 6 | [大模型炼丹术(六)：剖析 LLM 的解码策略](./docs/s1-foundations/剖析LLM的解码策略-大模型炼丹术(六).md) |
| 7 | [大模型炼丹术(七)：LLM 微调：训练一个垃圾邮件分类器](./docs/s1-foundations/LLM微调：训练一个垃圾邮件分类器-大模型炼丹术(七).md) |
| 8 | [大模型炼丹术(八)：LLM 指令微调：训练一个人工智能助手](./docs/s1-foundations/LLM指令微调：训练一个人工智能助手-大模型炼丹术(八).md) |

---

### 进阶篇 -- 《MiniMind 源码解读》 2️⃣

> 💡 [MiniMind](https://github.com/jingyaogong/minimind) 是一个轻量级大语言模型开源项目，代码规范易读，并且涵盖了现代 LLM 的核心技术点。进阶篇将通过对MiniMind源码解读的方式，深入 LLM 内部机制，讲解现代大语言模型的关键技术与优化方法。事实上，**可以将进阶篇看作是对于基础篇中一些未提及的核心概念的查漏补缺**。



学完进阶篇，你将掌握：

- 从零训练 tokenizer  
- RMSNorm 与模型归一化  
- 正余弦位置编码的局限与 RoPE 旋转位置编码  
- 注意力机制优化（GQA、MQA、KV Cache）  
- 稀疏模型 MoE  
- MiniMind 架构搭建  
- 自回归预训练、指令微调、DPO、LoRA 微调、LLM 蒸馏  

| 序号 | 文章链接 |
|------|----------|
| 1 | [MiniMind 源码解读（一）：如何从头训练 tokenizer](./docs/s2-advanced/1-如何从头训练tokenizer.md) |
| 2 | [MiniMind 源码解读（二）：一行代码之差，模型性能提升背后的 RMSNorm 玄机](./docs/s2-advanced/2-一行代码之差，模型性能提升背后的RMSNorm玄机.md) |
| 3 | [MiniMind 源码解读（三）：原始 Transformer 的位置编码及其缺陷](./docs/s2-advanced/3-原始Transformer的位置编码及其缺陷.md) |
| 4 | [MiniMind 源码解读（四）：旋转位置编码原理与应用全解析](./docs/s2-advanced/4-旋转位置编码原理与应用全解析.md) |
| 5 | [MiniMind 源码解读（五）：魔改注意力机制，细数当代 LLM 的效率优化手段](./docs/s2-advanced/5-魔改的注意力机制，细数当代LLM的效率优化手段.md) |
| 6 | [MiniMind 源码解读（六）：从稠密到稀疏，详解专家混合模型 MoE](./docs/s2-advanced/6-从稠密到稀疏，详解专家混合模型MOE.md) |
| 7 | [MiniMind 源码解读（七）：像搭积木一样构建一个大模型](./docs/s2-advanced/7-像搭积木一样构建一个大模型.md) |
| 8 | [MiniMind 源码解读（八）：LLM 预训练实践](./docs/s2-advanced/8-LLM预训练流程全解.md) |
| 9 | [MiniMind 源码解读（九）：指令微调详解，让大模型从“能说”变得“会听”](./docs/s2-advanced/9-指令微调详解-让大模型从“能说”变得“会听”.md) |
| 10 | [MiniMind 源码解读（十）：DPO - 大模型对齐训练新范式](./docs/s2-advanced/10-DPO-大模型对齐训练的新范式.md) |
| 11 | [MiniMind 源码解读（十一）：LoRA - LLM 轻量化微调利器](./docs/s2-advanced/11-LoRA-LLM轻量化微调的利器.md) |
| 12 | [MiniMind 源码解读（十二）：从白盒到黑盒，全面掌握大模型蒸馏技术](./docs/s2-advanced/12-从白盒到黑盒，全面掌握大模型蒸馏技术.md) |

---

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| 付修磊 | 项目负责人 | 坚信以输出驱动输入，并通过输出来深化理解和巩固知识 |


## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*

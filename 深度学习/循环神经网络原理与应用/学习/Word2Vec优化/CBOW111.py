import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
import re

# 数据预处理
corpus = [
    "jack like dog", "jack like cat", "jack like animal",
    "dog cat animal", "banana apple cat dog like", "dog fish milk like",
    "dog cat animal like", "jack like apple", "apple like", "jack like banana",
    "apple banana jack movie book music like", "cat dog hate", "cat dog like"
]

# 将句子分词，并转换为小写
def tokenize(sentence):
    # \b 单词的边界
    # \w+ 匹配一个或者多个单词字符（字母，数字，下划线）
    # \[,.!?] 匹配逗号、句号、感叹号和问号
    word_list = []
    for word in re.findall(r"\b\w+\b|[,.!?]", sentence):
        word_list.append(word.lower())
    return word_list


words = []
for sentence in corpus:
    for word in tokenize(sentence):
        words.append(word)

# print(words)
word_counts = Counter(words)

vocab = sorted(word_counts, key=word_counts.get, reverse=True)
# print(vocab)

# {"like": 1, "dog": 2}
# 创建词汇表到索引的隐射
vocab2int = {word: ii for ii, word in enumerate(vocab, 1)}
# print(vocab2int)
# 将所有的单词转换为索引表示
int2vocab = {ii: word for ii, word in enumerate(vocab, 1)}
# print(int2vocab)

# 将所有单词变成索引
word2index = [vocab2int[word] for word in words]
# print(word2index)
window = 1
center = []
context = []
negative_samples = []
num_negative_samples = 2
vocab_size = len(vocab2int) + 1  # 词汇表大小

for i, target in enumerate(word2index[window: -window], window):
    # print(i, target)
    # 数据：3 1 2 3 1 4
    # 索引：0 1 2 3 4 5
    center.append(target)
    con = word2index[i-window: i] + word2index[i+1: i+1+window]
    context.append(word2index[i-window: i] + word2index[i+1: i+1+window])
    # 生成负样本，确保不包括中心词和上下文
    negative_samples_i = []
    for _ in range(num_negative_samples):
        negative_sample = np.random.choice(vocab_size)
        while negative_sample == target or negative_sample in con:
            negative_sample = np.random.choice(vocab_size)
        negative_samples_i.append(negative_sample)
    negative_samples.append(negative_samples_i)


torch.manual_seed(10)

# 特殊标记：0，用于填充或者标记未知单词
# <SOS>: 句子起始标识符
# <EOS>：句子结束标识符
# <PAD>：补全字符
# <MASK>：掩盖字符
# <SEP>：两个句子之间的分隔符
# <UNK>：低频或未出现在词表中的词
# vocab_size = len(vocab2int) + 1  # 词汇表大小
embedding_dim = 2  # 嵌入维度


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        # self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))  # 初始化embedding矩阵
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 初始化embedding矩阵
        self.linear = nn.Linear(embedding_dim, vocab_size)  # vocab_size = 14

    def forward(self, context):
        # context.shape    ->   [bs, 2]
        context_emb = self.embeddings(context)  # 上下文的嵌入向量
        # print(context_emb.shape)  # torch.Size([4, 2, 2])
        avg_emb = torch.mean(context_emb, dim=1, keepdim=True).squeeze(1)
        y = self.linear(avg_emb)
        return y

model = CBOWModel(vocab_size, embedding_dim)
cri = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


bs = 4
epochs = 2000
for epoch in range(1, epochs+1):
    total_loss = 0
    for batch_index in range(0, len(context), bs):
        # 上下文的Tensor  # torch.Size([4, 2])
        context_tensor = torch.tensor(context[batch_index: batch_index+bs])
        # 中心词的tensor  torch.Size([4, 1])
        center_tensor = torch.tensor(center[batch_index: batch_index+bs]).view(bs, 1)
        # 负样本的tensor  torch.Size([4, 2])
        negative_tensor = torch.tensor(negative_samples[batch_index: batch_index+bs])

        # 正样本
        # 第一维 4 表示批次中的样本数。第二维 2 表示上下文窗口的大小，即每个中心词有两个上下文词。第三维 2 表示嵌入向量的维度（embedding dimension）。
        # print("context_emb", context_emb.shape)  # torch.Size([4, 2, 2])
        context_emb = model.embeddings(context_tensor)

        # 第一维 4 表示批次中的样本数。第二维 2 表示嵌入向量的维度。
        # print("center_emb", center_emb.shape)  # torch.Size([4, 2])
        center_emb = model.embeddings(center_tensor).squeeze(1)

        # 将多个上下文词的嵌入向量平均化，从而得到每个样本的平均上下文嵌入向量。这有助于简化计算，并能更好地表示上下文的总体信息。
        avg_context_emb = torch.mean(context_emb, dim=1)  # avg_context_emb 的形状是 [4, 2]
        """
        例如：
        context_emb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]],
                            [[0.5, 0.6], [0.7, 0.8]],
                            [[0.9, 1.0], [1.1, 1.2]],
                            [[1.3, 1.4], [1.5, 1.6]]])
        通过执行 avg_context_emb = torch.mean(context_emb, dim=1)，我们在 context_window_size 维度上取平均值：
        对第一个样本：[(0.1 + 0.3)/2, (0.2 + 0.4)/2] = [0.2, 0.3]
        对第二个样本：[(0.5 + 0.7)/2, (0.6 + 0.8)/2] = [0.6, 0.7]
        对第三个样本：[(0.9 + 1.1)/2, (1.0 + 1.2)/2] = [1.0, 1.1]
        对第四个样本：[(1.3 + 1.5)/2, (1.4 + 1.6)/2] = [1.4, 1.5]
        avg_context_emb = torch.tensor([[0.2, 0.3],
                                [0.6, 0.7],
                                [1.0, 1.1],
                                [1.4, 1.5]])
        avg_context_emb 的形状变为 [4, 2]，2表示每个样本的平均上下文嵌入向量。
        """

        positive_scores = torch.matmul(avg_context_emb, center_emb.t())
        # 第一维 4 批次中的样本数（batch size）。第二维 4 与批次中的每一个中心词的相似性分数。
        # positive_scores[i, j] 表示第 i 个样本的平均上下文嵌入与第 j 个样本的中心词嵌入之间的相似性分数。
        # print("positive_scores", positive_scores.shape)  # torch.Size([4, 4])
        # positive_labels = torch.ones_like(positive_scores)
        positive_labels = torch.eye(bs)  # 生成对角线为1的标签矩阵
        positive_loss = cri(positive_scores, positive_labels)

        # 负样本
        # negative_emb.shape  [batch_size, num_negative_samples, embedding_dim]:[4, 2, 2]
        negative_emb = model.embeddings(negative_tensor)
        # avg_context_emb->[batch_size, embedding_dim]   avg_context_emb.unsqueeze(1)->[batch_size, 1, embedding_dim]
        # negative_emb.permute(0, 2, 1)  ->  [batch_size, embedding_dim, num_negative_samples]
        # torch.matmul() ->  [batch_size, 1, num_negative_samples]
        # 通过 squeeze(1) 去掉维度1，得到 negative_scores 的形状是 [batch_size, num_negative_samples]。
        # 每个中心词与其负样本（negative）之间的相似性
        negative_scores = torch.matmul(avg_context_emb.unsqueeze(1), negative_emb.permute(0, 2, 1)).squeeze(1)
        negative_labels = torch.zeros_like(negative_scores)
        negative_loss = cri(negative_scores, negative_labels)

        loss = positive_loss + negative_loss
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(context)
    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss {avg_loss:.4f}")
        # 每个单词的embedding向量
        word_vec = model.embeddings.weight.detach().numpy()
        # print(word_vec)
        x = word_vec[:, 0]
        y = word_vec[:, 1]
        selected_word = ["dog", "cat", "milk", "like", "animal", "fish", "banana", "apple"]
        selected_word_index = [vocab2int[word] for word in selected_word]
        selected_word_x = x[selected_word_index]
        selected_word_y = y[selected_word_index]
        plt.cla()
        plt.scatter(selected_word_x, selected_word_y, color="blue")
        # 将每个点的标注加上
        for word, x, y in zip(selected_word, selected_word_x, selected_word_y):
            plt.annotate(word, (x, y), textcoords="offset points", xytext=(0, 10))
        plt.pause(0.5)

plt.show()
















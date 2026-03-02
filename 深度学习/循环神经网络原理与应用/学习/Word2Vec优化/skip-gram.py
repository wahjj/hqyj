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
    word_list = []
    for word in re.findall(r"\b\w+\b|[,.!?]", sentence):
        word_list.append(word.lower())
    return word_list

words = []
for sentence in corpus:
    for word in tokenize(sentence):
        words.append(word)

word_counts = Counter(words)

vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# 创建词汇表到索引的隐射
vocab2int = {word: ii for ii, word in enumerate(vocab, 1)}
int2vocab = {ii: word for ii, word in enumerate(vocab, 1)}

# 将所有单词变成索引
word2index = [vocab2int[word] for word in words]
window = 1
center = []
context = []
negative_samples = []
num_negative_samples = 2
vocab_size = len(vocab2int) + 1  # 词汇表大小

for i, target in enumerate(word2index[window: -window], window):
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

embedding_dim = 2  # 嵌入维度

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center):
        center_emb = self.embeddings(center)
        y = self.linear(center_emb)
        return y

model = SkipGramModel(vocab_size, embedding_dim)
cri = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

bs = 4
epochs = 2000
for epoch in range(1, epochs+1):
    total_loss = 0
    for batch_index in range(0, len(center), bs):
        center_tensor = torch.tensor(center[batch_index: batch_index+bs]).view(bs, 1)
        context_tensor = torch.tensor(context[batch_index: batch_index+bs])
        negative_tensor = torch.tensor(negative_samples[batch_index: batch_index+bs])


        # 在 CBOW 模型中，计算的是平均上下文嵌入与中心词嵌入之间的相似性，得到的 positive_scores 的形状是 [batch_size, batch_size]。
        # 在 Skip-gram 模型中，计算的是每个中心词与其上下文词之间的相似性，得到的 positive_scores 的形状是 [batch_size, num_context_samples]。
        # 正样本
        # 第一维 4 表示批次中的样本数。第二维 2 表示上下文窗口的大小，即每个中心词有两个上下文词。第三维 2 表示嵌入向量的维度（embedding dimension）。
        # print("context_emb", context_emb.shape)  # torch.Size([4, 2, 2])
        context_emb = model.embeddings(context_tensor)

        # 第一维 4 表示批次中的样本数。第二维 2 表示嵌入向量的维度。
        # print("center_emb", center_emb.shape)  # torch.Size([4, 2])
        center_emb = model.embeddings(center_tensor).squeeze(1)
        # center_emb->[batch_size, embedding_dim]   center_emb.unsqueeze(1)->[batch_size, 1, embedding_dim]
        # context_emb.permute(0, 2, 1)  ->  [batch_size, embedding_dim, num_context_samples]
        # torch.matmul() ->  [batch_size, 1, num_context_samples]
        # 通过 squeeze(1) 去掉维度1，得到 negative_scores 的形状是 [batch_size, num_context_samples]
        positive_scores = torch.matmul(center_emb.unsqueeze(1), context_emb.permute(0, 2, 1)).squeeze(1)
        print("positive_scores:", positive_scores.shape)
        positive_labels = torch.ones_like(positive_scores)
        positive_loss = cri(positive_scores, positive_labels)

        # 负样本
        # negative_emb.shape  [batch_size, num_negative_samples, embedding_dim]:[4, 2, 2]
        negative_emb = model.embeddings(negative_tensor)
        # center_emb->[batch_size, embedding_dim]   center_emb.unsqueeze(1)->[batch_size, 1, embedding_dim]
        # negative_emb.permute(0, 2, 1)  ->  [batch_size, embedding_dim, num_negative_samples]
        # torch.matmul() ->  [batch_size, 1, num_negative_samples]
        # 通过 squeeze(1) 去掉维度1，得到 negative_scores 的形状是 [batch_size, num_negative_samples]
        negative_scores = torch.matmul(center_emb.unsqueeze(1), negative_emb.permute(0, 2, 1)).squeeze(1)
        negative_labels = torch.zeros_like(negative_scores)
        negative_loss = cri(negative_scores, negative_labels)

        loss = positive_loss + negative_loss
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(center)
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

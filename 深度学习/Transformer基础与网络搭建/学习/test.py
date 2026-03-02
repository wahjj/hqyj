import torch

trg_subseq_mask = torch.tril(torch.ones(10, 10)).bool()
print(trg_subseq_mask)
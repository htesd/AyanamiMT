from  AyanamiiMT import AyanamiMT1_3
import torch
# 初始化模型 (参数示例)
model = AyanamiMT1_3(
    max_lens=200,  # 单块处理长度
    memory_length=64,  # 记忆单元长度
    vocab_size=50304,  # 根据实际词表调整
    num_layer=6,  # 层数示例
    batch_size=2
)

x=torch.abs(torch.randn(2, 200))

x=x.to(torch.long)

print(x.shape)

result=model(x)

print(result.shape)
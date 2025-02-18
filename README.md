
```markdown
# Ayanami-RMT (实验性实现)

基于Transformer的循环记忆增强实现，探索长序列处理的可能性

## 快速使用

### 安装依赖
```bash
pip install torch>=2.0.0
```

### 基础推理
```python
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
```

### 最小训练示例
```python
from torch.optim import AdamW
from  AyanamiiMT import AyanamiMT1_3
import torch
# 训练配置
config = {
    'batch_size': 2,
    'context_window': 4096,
    'lr': 3e-4,
    'grad_accum_steps': 4
}

# 初始化
model = AyanamiMT1_3(max_lens=config['context_window'], memory_length=64)
optimizer = AdamW(model.parameters(), lr=config['lr'])

# 简化训练循环
for step in range(10):
    optimizer.zero_grad()

    # 梯度累积
    for micro_step in range(config['grad_accum_steps']):
        inputs, targets = get_batch()  # 需实现数据加载

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            loss /= config['grad_accum_steps']

        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

## 关键参数说明
```python
# 模型初始化参数
AyanamiMT1_3(
    max_lens=2048,       # 单块最大处理长度 (建议1024-2048)
    memory_length=64,    # 记忆token数量 (推荐32-128)
    d_model=768,         # 隐藏层维度
    num_head=8,          # 注意力头数
    num_layer=6,         # 层深
    dropout=0.1          # 防止过拟合
)
```

## 使用建议
1. **长序列处理**：按`max_lens`分块处理，保留记忆状态
2. **内存优化**：结合`torch.utils.checkpoint`使用
3. **精度控制**：推荐使用BF16混合精度
4. **批处理**：小批量+梯度累积效果更好

## 注意事项
- 本实现为实验性质，尚未充分验证
- 实际性能可能随参数配置变化
- 建议从small-scale配置开始实验
```


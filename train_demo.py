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
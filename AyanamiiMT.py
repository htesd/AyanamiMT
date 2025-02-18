import math
import torch
from sympy import false
from torch import nn
from matplotlib import pyplot as plt
# from my_lora import LoRALayer, Linear_lora, Linear_add as Linear
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# from nets import LoRALayer

'''
修改理念：保证参数命名的一致性，最主要的就是lens和dim
用于方便进行类型转换和lora注入
'''
model_dtype = torch.bfloat16


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class DotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout_rate).to(model_dtype)

    def forward(self, q, k, v, mask=None, valid_lens=None):

        batch_size, num_head, lens_q, d = q.shape[-4:]
        scale = 1 / math.sqrt(d)
        score = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)

            score = score + mask

        if valid_lens is not None:
            '''
            生成有效长度注意力
            '''

            idxs = torch.arange(0, lens_q, device=valid_lens.device).unsqueeze(0).expand(batch_size, -1)
            mask = (idxs >= valid_lens.unsqueeze(1))
            expanded_mask = mask.unsqueeze(1).unsqueeze(-1)
            expanded_mask = (expanded_mask | expanded_mask.transpose(2, 3)).float() * -1e9
            score += expanded_mask

        w = F.softmax(score, dim=-1)
        w = self.dropout(w)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class MultiHeadAttention(nn.Module):

    def __init__(self, feature_size, num_head, dropout=None, bias=False, q_size=None, k_size=None, v_size=None,
                 **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.num_head = num_head
        if dropout == None:
            self.attention = DotProductAttention()
        else:
            self.attention = DotProductAttention(dropout_rate=dropout)

        if q_size == None:
            q_size = feature_size
        if k_size == None:
            k_size = feature_size
        if v_size == None:
            v_size = feature_size

        '''
        !TODO: 这里我觉得全链接层也没有用啊，我你吗。到时候去了看看
        '''
        self.wq = Linear(q_size, feature_size)
        self.wk = Linear(k_size, feature_size, bias=False)
        self.wv = Linear(v_size, feature_size)
        self.wo = Linear(feature_size, feature_size)

    def forward(self, q, k, v, mask=None, valid_lens=None):

        q = self.transpose_qkv(self.wq(q), self.num_head)

        k = self.transpose_qkv(self.wk(k), self.num_head)

        v = self.transpose_qkv(self.wv(v), self.num_head)

        output = self.attention(q, k, v, mask, valid_lens)

        return self.wo(output)

    def transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads， num_hiddens/num_heads)
        X = X.view(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        return X


class PositionWiseFFN(nn.Module):
    def __init__(self, num_input, num_hiddens, num_outputs, **kwargs):
        '''
        :param 这里个人认为以后可以尝试吧linear换成卷积
        2023年11月，现在已经一年过去了
        '''
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.linear1 = Linear(num_input, num_hiddens)
        self.gelu = nn.GELU()
        self.linear2 = Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)

        return x


class PositionWiseFFN_Lora(nn.Module):
    def __init__(self, num_input, num_hiddens, num_outputs, layer_sequence, lora_dim, **kwargs):
        '''
        :param 这里个人认为以后可以尝试吧linear换成卷积
        2023年11月，现在已经一年过去了
        2024年12月17日，今天继续改进
        暂时这个地方loradim只能为16
        '''

        super(PositionWiseFFN_Lora, self).__init__(**kwargs)

        self.linear1 = Linear(num_input, num_hiddens)
        self.gelu = nn.GELU()
        self.linear2 = Linear(num_hiddens, num_outputs)

        self.num_hiddens = num_hiddens
        self.num_input = num_input
        self.num_outputs = num_outputs
        self.lora_dim = lora_dim
        self.layer_sequence = layer_sequence
        self.lora_token_index = self.lora_dim * layer_sequence + 1

        ##   和learning rate一个量级，这是为什么呢？
        self.scale = nn.Parameter(torch.tensor(0.001))

    def forward(self, x):
        '''
        这个地方只其他不需要经过lora的也无所谓的，同样需要经过处理
        但是这里需要做一个归一化函数，越往前的层理论上应该受token的影响就越小，所以这里需要一个归一化函数保证总影响为1
        这个地方可以复用memorytoken，但是需要做实验,个人觉得效果不会太好
        不对啊，我有这个以后我还需要memory吗？
        或许不需要啊！
        所以我需要做三个实验。
        '''
        batch_size, seq_len, _ = x.shape

        # 1. 提取 LoRA tokens
        lora_token_this_layer = x[:, self.lora_token_index:self.lora_token_index + self.lora_dim, :]

        # with torch.no_grad():
        #     print(self.layer_sequence)
        #     print(self.scale)
        #     print(lora_token_this_layer.mean())

        even_tokens = lora_token_this_layer[:, ::2, :]  # [batch_size, lora_dim/2, num_input]
        odd_tokens = lora_token_this_layer[:, 1::2, :]  # [batch_size, lora_dim/2, num_input]

        result_multiplication = torch.matmul(even_tokens.unsqueeze(-1), odd_tokens.unsqueeze(-2))
        matrix1 = result_multiplication[:, :int(result_multiplication.shape[1] / 2), :, :]
        matrix1 = matrix1.view(batch_size, -1, self.num_input)
        temp_weight1 = self.linear1.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        temp_weight1 = temp_weight1 + matrix1 * self.scale
        x = x @ temp_weight1.transpose(-1, -2) + self.linear1.bias

        x = self.gelu(x)

        matrix2 = result_multiplication[:, -int(result_multiplication.shape[1] / 2):, :, :]
        matrix2 = matrix2.view(batch_size, self.num_input, -1)
        temp_weight2 = self.linear2.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        temp_weight2 = temp_weight2 + matrix2 * self.scale
        x = x @ temp_weight2.transpose(-1, -2) + self.linear2.bias

        return x


'''
test code 
为了并行lora同样需要拷贝两份，前面一份后面一份这样才不会导致信息泄露
但是不对啊，我就是想泄露信息
如果我想简单的实现那就得拷贝两份，但是如果我想只使用一个来进行我就需要保证transformer每个层使用的mask是不一样的,这里还是从简
暂时定lora必须为16的倍数这里取16*lora的倍数
'''

lora_dim = 16 * 1

input = torch.randn([3, 100, 512])

FFN_LORA = PositionWiseFFN_Lora(512, 512 * 4, 512, 1, lora_dim)

print(FFN_LORA(input).shape)


class AddNorm(nn.Module):
    def __init__(self, shape, dropout_rate=0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)

        self.ln = LayerNorm(shape)

    def forward(self, X, Y):
        # 这地方非常重要，如果相加再norm，那效果就会差很多，尤其是在长上下文下，
        # 我之前怎么没发现这个问题？？？？
        # woc？？？？？？？？？ 为什么我相加再合并在rnn上反而表现好啊？
        # 后面还是需要重新测试下
        return self.ln(X + Y)


class AyanamiT2DecoderLayer(nn.Module):
    def __init__(self, num_hidden, num_head, q_size, k_size, v_size, ffn_num_input, ffn_num_hidden, ffn_num_output, i,
                 dropout=0.0, use_bias=True, use_lora_token=false):
        super().__init__()

        self.attention = MultiHeadAttention(feature_size=num_hidden, num_head=num_head, q_size=q_size,
                                            k_size=k_size, v_size=v_size, dropout=dropout, bias=use_bias)
        self.layer_norm1 = LayerNorm(num_hidden)
        if use_lora_token:
            self.ffn = PositionWiseFFN_Lora(ffn_num_input, ffn_num_hidden, ffn_num_output, layer_sequence=i,
                                            lora_dim=16)
        else:
            self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, ffn_num_output)
        self.layer_norm2 = LayerNorm(num_hidden)

    def forward(self, X, mask=None, state=None, valid_lens=None):

        Xn = self.layer_norm1(X)

        X = X + self.attention(Xn, Xn, Xn, mask, valid_lens)

        X = X + self.ffn(self.layer_norm2(X))

        return X, None


class AyanamiT3DecoderLayer(nn.Module):
    '''
    这个解码层被设计用来生成层lora，用于把lora token和现有的层合并，
    lora token 和常规token以及memorytoken处理之后，再提取出来为下一次循环构成计算图
    目前暂时把所有的lora token放在最后,进行lora操作后能够和一个特定的ffn相加

    '''

    def __init__(self, num_hidden, num_head, q_size, k_size, v_size, ffn_num_input, ffn_num_hidden, ffn_num_output, i,
                 dropout=0.0, use_bias=True):
        super().__init__()

        self.attention = MultiHeadAttention(feature_size=num_hidden, num_head=num_head, q_size=q_size,
                                            k_size=k_size, v_size=v_size, dropout=dropout, bias=use_bias)
        self.layer_norm1 = LayerNorm(num_hidden)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, ffn_num_output)
        self.layer_norm2 = LayerNorm(num_hidden)

    def forward(self, X, mask=None, state=None, valid_lens=None):
        Xn = self.layer_norm1(X)

        X = X + self.attention(Xn, Xn, Xn, mask, valid_lens)

        X = X + self.ffn(self.layer_norm2(X))

        return X, None


class AyanamiT2(nn.Module):

    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.001,
                 bias=True):

        super().__init__()
        self.max_lens = max_lens
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)
        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)
        self.drop = nn.Dropout(dropout)
        # mask.requires_grad_(false)

        self.register_buffer("mask", mask, persistent=False)

        self.bloks = nn.Sequential()
        for i in range(num_layer):
            self.bloks.add_module("AyanamiT2DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,

                                                        use_bias=bias))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        x = self.embedding(x)
        x = x.to(model_dtype)

        b, t, e = x.shape
        assert t <= self.max_lens, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # (t)
        pos_emb = self.pos_embedding(pos)  # (t, n_embd)
        x = self.drop(x + pos_emb)  # (b, t, n_embd)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        result = (
                x @ torch.transpose(self.embedding.weight.to(x.dtype), 0, 1)
        )

        print(self.embedding.weight.shape)
        print(result.shape)

        return result


class AyanamiMT1(nn.Module):
    '''
    这个模型为了memory token设计，
    该模型把token试做一个memory单元，并且利用checkpoint来实现长序列训练
    这个地方逻辑有问题，memory可以看到后面所有的token，同时token又可以看到memory，这违背了因果推理的初衷，
    理论上最后loss应该会降低到0
    如果有一天，真的可以是有一个和人一样的ai就好了，但是我真的很累了
    '''

    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.001,
                 memory_length=108, batch_size=1, bias=True):

        super().__init__()

        self.memory_length = memory_length
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)

        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)

        memory_mask = torch.empty(max_lens + memory_length + 2, max_lens + memory_length + 2).fill_(0).to(model_dtype)

        memory_mask[memory_length + 2:, memory_length + 2:] += mask

        self.memory_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        self.memory = nn.Parameter(torch.empty(batch_size, memory_length, d_model))
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.memory_token_embedding)

        memory_mask.requires_grad_(False)
        self.register_buffer("mask", memory_mask, persistent=False)

        self.bloks = nn.Sequential()
        for i in range(num_layer):
            self.bloks.add_module("AyanamiMMT1DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,

                                                        use_bias=bias))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        result, new_memory = checkpoint.checkpoint(
            self.train_forward,
            x,
            self.memory,
            preserve_rng_state=True
        )

        self.memory.data = (self.memory.data + new_memory) / 2

        return result

    def train_forward(self, x, memory):
        '''
        :param x:
        :return:
        '''

        x = self.embedding(x)

        batch_size = x.shape[0]

        memory_token = self.memory_token_embedding.expand(batch_size, -1, -1)
        x = x.to(model_dtype)

        x = torch.cat([memory_token, memory, memory_token, x], dim=1)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        new_memory = x[:, 1:1 + self.memory_length, :]

        result = (
                x[:, 2 + self.memory_length:, :] @ torch.transpose(self.embedding.weight.to(x.dtype), 0, 1)
        )

        return result, new_memory


class AyanamiMT1_1(nn.Module):
    '''
    这个模型为了memory token设计，
    该模型把token试做一个memory单元，并且利用checkpoint来实现长序列训练
    对memory加入了区分，前面的是输入，后面的是输出
    old memory
    full[:2,2:]=-1

    full[2:4,-2:]=-1
    casual
    full[2:4,2:4].fill_(-1).triu_(1)
    tensor([[ 0.,  0., -1., -1., -1., -1.],
            [-0.,  0., -1., -1., -1., -1.],
            [-0.,  0.,  0., -1., -1., -1.],
            [-0.,  0.,  0.,  0., -1., -1.],
            [-0.,  0., -0.,  0., -0.,  0.],
            [-0.,  0., -0.,  0., -0.,  0.]])
    '''

    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.000,
                 memory_length=108, batch_size=1, bias=True):

        super().__init__()

        self.max_lens = max_lens
        self.memory_length = memory_length
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)
        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)
        memory_mask = torch.empty(max_lens + 2 * (memory_length + 2), max_lens + 2 * (memory_length + 2)).fill_(0).to(
            model_dtype)
        # casual mask
        memory_mask[memory_length + 2:-memory_length - 2, memory_length + 2:-memory_length - 2] += mask
        # old memory mask
        memory_mask[:memory_length + 2, memory_length + 2:] = -torch.inf
        memory_mask[memory_length + 2:(memory_length + 2) + max_lens, -memory_length - 2:] = -torch.inf

        self.memory_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        self.memory = nn.Parameter(torch.empty(batch_size, memory_length, d_model))
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.memory_token_embedding)

        memory_mask.requires_grad_(False)
        self.register_buffer("mask", memory_mask, persistent=False)

        self.bloks = nn.Sequential()
        for i in range(num_layer):
            self.bloks.add_module("AyanamiMMT1DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,

                                                        use_bias=bias))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        y_hat_list = []

        block_times = int(x.shape[1] / self.max_lens) + 1

        for i in range(block_times):
            result, new_memory = checkpoint.checkpoint(
                self.train_forward,
                x[:, self.max_lens * i:self.max_lens * i + self.max_lens],
                self.memory,
                preserve_rng_state=True
            )
            y_hat_list.append(result)

            self.memory.data = (self.memory.data + new_memory) / 2

        return torch.cat(y_hat_list, dim=1)

    def train_forward(self, x, memory):

        x = self.embedding(x)

        batch_size = x.shape[0]

        memory_token = self.memory_token_embedding.expand(batch_size, -1, -1)

        x = x.to(model_dtype)

        x = torch.cat([memory_token, memory, memory_token, x, memory_token, memory, memory_token], dim=1)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        new_memory = x[:, -1 - self.memory_length:-1, :]

        result = (
                x[:, 2 + self.memory_length:-2 - self.memory_length, :] @ torch.transpose(
            self.embedding.weight.to(x.dtype), 0, 1)
        )

        return result, new_memory


class AyanamiMT1_2(nn.Module):
    '''
    这个模型为了memory token设计，
    该模型把token试做一个memory单元，并且利用checkpoint来实现长序列训练
    对memory加入了区分，前面的是输入，后面的是输出
    之前的模型出现了很严重的波动问题，我需要想办法稳定这种情况,这个地方决定采用非学习的中间状态
    '''

    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.000,
                 memory_length=108, batch_size=1, bias=True):

        super().__init__()

        self.max_lens = max_lens
        self.memory_length = memory_length
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)

        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)
        memory_mask = torch.empty(max_lens + 2 * (memory_length + 2), max_lens + 2 * (memory_length + 2)).fill_(0).to(
            model_dtype)
        # casual mask
        memory_mask[memory_length + 2:-memory_length - 2, memory_length + 2:-memory_length - 2] += mask
        # old memory mask
        memory_mask[:memory_length + 2, memory_length + 2:] = -torch.inf
        memory_mask[memory_length + 2:(memory_length + 2) + max_lens, -memory_length - 2:] = -torch.inf

        self.memory_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        self.memory = nn.Parameter(torch.empty(batch_size, memory_length, d_model).fill_(0.0))

        nn.init.xavier_uniform_(self.memory_token_embedding)
        nn.init.xavier_uniform_(self.memory)

        memory_mask.requires_grad_(False)
        self.register_buffer("mask", memory_mask, persistent=False)

        self.bloks = nn.Sequential()
        for i in range(num_layer):
            self.bloks.add_module("AyanamiMMT1DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,

                                                        use_bias=bias))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        y_hat_list = []
        #这个地方我为什么要+1呢？
        block_times = int(x.shape[1] / self.max_lens) + 1

        self.memory.data = (self.memory.data * 0.0) + 0.001

        for i in range(block_times):
            result, new_memory = checkpoint.checkpoint(
                self.train_forward,
                x[:, self.max_lens * i:self.max_lens * i + self.max_lens],
                self.memory,
                preserve_rng_state=True
            )
            y_hat_list.append(result)

            self.memory.data = (self.memory.data + new_memory) / 2

        return torch.cat(y_hat_list, dim=1)

    def train_forward(self, x, memory):

        x = self.embedding(x)

        batch_size = x.shape[0]

        memory_token = self.memory_token_embedding.expand(batch_size, -1, -1)

        x = x.to(model_dtype)

        x = torch.cat([memory_token, memory, memory_token, x, memory_token, memory, memory_token], dim=1)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        new_memory = x[:, -1 - self.memory_length:-1, :]

        result = (
                x[:, 2 + self.memory_length:-2 - self.memory_length, :] @ torch.transpose(
            self.embedding.weight.to(x.dtype), 0, 1)
        )

        return result, new_memory


class AyanamiMT1_3(nn.Module):
    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.000,
                 memory_length=108, batch_size=1, bias=True):
        super().__init__()
        '''
            tensor(
           [[ 0.,  0., -1., -1., -1., -1.],
            [-0.,  0., -1., -1., -1., -1.],
            [-0.,  0.,  0., -1., -1., -1.],
            [-0.,  0.,  0.,  0., -1., -1.],
            [-0.,  0., -0.,  0., -0.,  0.],
            [-0.,  0., -0.,  0., -0.,  0.]])
        '''

        '''
        论文点子

        理论支持：ttt
        让memory和模型参数接受输入参数的洗礼！
        
        这个和t2有什么区别？
        
        '''

        self.max_lens = max_lens
        self.memory_length = memory_length
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)
        # casual
        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)

        memory_mask = torch.empty(max_lens + 2 * (memory_length + 2), max_lens + 2 * (memory_length + 2)).fill_(0).to(
            model_dtype)
        # casual
        memory_mask[self.memory_length + 2:-self.memory_length - 2,
        self.memory_length + 2:-self.memory_length - 2] += mask
        # old memory mask
        memory_mask[:self.memory_length + 2, self.memory_length + 2:] = -torch.inf
        memory_mask[self.memory_length + 2:(self.memory_length + 2) + self.max_lens,
        -self.memory_length - 2:] = -torch.inf

        self.memory_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))

        # 将memory设为模型参数而非缓冲区
        '''
        这个地方还是需要思考下关于不同batch的问题还是应该在一开始创建一个简单memory，然后在迭代的时候再复制
        '''
        self.memory = nn.Parameter(torch.empty(batch_size, memory_length, d_model))

        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.memory_token_embedding)
        memory_mask.requires_grad_(False)
        self.register_buffer("mask", memory_mask, persistent=False)
        self.bloks = nn.Sequential()
        for i in range(num_layer):
            self.bloks.add_module("AyanamiMMT1DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,
                                                        use_bias=bias))
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        y_hat_list = []
        # 这个地方不一定要加1,逻辑还是比较复杂，需要考虑除尽和除不尽两种情况
        if x.shape[1] % self.max_lens == 0:
            block_times = int(x.shape[1] / self.max_lens)
        else:

            block_times = int(x.shape[1] / self.max_lens) + 1
            print("error!!!! 当前代码没有考虑除不尽的情况！")

        # 克隆memory以避免原地修改
        memory = self.memory

        for i in range(block_times):
            result, new_memory = checkpoint.checkpoint(
                self.train_forward,
                x[:, self.max_lens * i:self.max_lens * i + self.max_lens],
                memory,
                preserve_rng_state=True
            )
            y_hat_list.append(result)

            # 更新memory，不使用detach()
            # 待修改
            memory = (memory + new_memory) / 2

        # 更新模型的memory参数
        # self.memory = nn.Parameter(memory.detach())

        return torch.cat(y_hat_list, dim=1)

    def train_forward(self, x, memory):
        # 这个可能不需要放在里面
        x = self.embedding(x)

        b, t, e = x.shape
        assert t <= self.max_lens, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # (t)
        pos_emb = self.pos_embedding(pos)  # (t, n_embd)
        x = x + pos_emb  # (b, t, n_embd)
        x = x.to(model_dtype)

        memory_token = self.memory_token_embedding.expand(b, -1, -1)

        x = torch.cat([memory_token, memory, memory_token, x, memory_token, memory, memory_token], dim=1)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        new_memory = x[:, -1 - self.memory_length:-1, :]

        # 这个地方最终会产生一个很大的矩阵，这没办法
        # 100000*50

        result = (
                x[:, 2 + self.memory_length:-2 - self.memory_length, :] @ torch.transpose(
            self.embedding.weight.to(x.dtype), 0, 1)
        )

        return result, new_memory


class AyanamiMT1_4(nn.Module):
    def __init__(self, max_lens=1024, vocab_size=50304, d_model=768, num_layer=12, num_head=12, dropout=0.000,
                 memory_length=108, batch_size=1, lora_dim=1, bias=True):
        '''

        :param max_lens:
        :param vocab_size:
        :param d_model:
        :param num_layer:
        :param num_head:
        :param dropout:
        :param memory_length:
        :param batch_size:
        :param bias:
        尝试把lora融入token中，让lora和并入模型参数中，让整个模型都不停的更新！

        '''
        super().__init__()
        '''
            tensor(
           [[ 0.,  0., -1., -1., -1., -1.],
            [-0.,  0., -1., -1., -1., -1.],
            [-0.,  0.,  0., -1., -1., -1.],
            [-0.,  0.,  0.,  0., -1., -1.],
            [-0.,  0., -0.,  0., -0.,  0.],
            [-0.,  0., -0.,  0., -0.,  0.]])
        '''

        '''
        论文点子
        理论支持：ttt
        让memory和模型参数接受输入参数的洗礼！
        主要目的是为了更新ffn层，还有attention层
        需要做三次实验：1只更新attention 2只更新ffn 3都更新

        还得考虑：把memory和ffn混用，分开使用
        '''

        self.max_lens = max_lens
        self.memory_length = memory_length
        self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)
        self.pos_embedding = nn.Embedding(max_lens, d_model).to(model_dtype)
        # casual
        mask = torch.empty(max_lens, max_lens).fill_(-torch.inf).triu_(1).to(model_dtype)

        memory_mask = torch.empty(max_lens + 2 * (memory_length + 2), max_lens + 2 * (memory_length + 2)).fill_(0).to(
            model_dtype)
        # casual
        memory_mask[self.memory_length + 2:-self.memory_length - 2,
        self.memory_length + 2:-self.memory_length - 2] += mask
        # old memory mask
        memory_mask[:self.memory_length + 2, self.memory_length + 2:] = -torch.inf
        memory_mask[self.memory_length + 2:(self.memory_length + 2) + self.max_lens,
        -self.memory_length - 2:] = -torch.inf

        self.memory_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        # self.lora_token_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        # 将memory设为模型参数而非缓冲区 这个地方暂时不把loratoken加入
        self.memory = nn.Parameter(torch.empty(batch_size, memory_length, d_model))
        # self.lora_token = nn.Parameter(torch.empty(batch_size, num_layer, d_model))

        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.memory_token_embedding)
        memory_mask.requires_grad_(False)
        self.register_buffer("mask", memory_mask, persistent=False)
        self.bloks = nn.Sequential()

        for i in range(num_layer):
            self.bloks.add_module("AyanamiMMT1DecoderLayer" + str(i),
                                  AyanamiT2DecoderLayer(d_model, num_head, d_model, d_model, d_model, d_model,
                                                        d_model * 4,
                                                        d_model, i=i,
                                                        dropout=dropout,
                                                        use_bias=bias,
                                                        use_lora_token=True
                                                        ))
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        y_hat_list = []
        # 这个地方不一定要加1,逻辑还是比较复杂，需要考虑除尽和除不尽两种情况
        if x.shape[1] % self.max_lens == 0:
            block_times = int(x.shape[1] / self.max_lens)
        else:

            block_times = int(x.shape[1] / self.max_lens) + 1
            print("error!!!! 当前代码没有考虑除不尽的情况！")

        # 克隆memory以避免原地修改
        memory = self.memory

        for i in range(block_times):
            result, new_memory = checkpoint.checkpoint(
                self.train_forward,
                x[:, self.max_lens * i:self.max_lens * i + self.max_lens],
                memory,
                preserve_rng_state=True
            )
            y_hat_list.append(result)

            # 更新memory，不使用detach()
            # 待修改
            memory = (memory + new_memory) / 2

        # 更新模型的memory参数
        # self.memory = nn.Parameter(memory.detach())

        return torch.cat(y_hat_list, dim=1)

    def train_forward(self, x, memory):
        # 这个可能不需要放在里面
        x = self.embedding(x)

        b, t, e = x.shape
        assert t <= self.max_lens, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # (t)
        pos_emb = self.pos_embedding(pos)  # (t, n_embd)
        x = x + pos_emb  # (b, t, n_embd)
        x = x.to(model_dtype)

        memory_token = self.memory_token_embedding.expand(b, -1, -1)

        x = torch.cat([memory_token, memory, memory_token, x, memory_token, memory, memory_token], dim=1)

        for i, blk in enumerate(self.bloks):
            x, temp_state = blk(x, self.mask, None, None)

        new_memory = x[:, -1 - self.memory_length:-1, :]

        # 这个地方最终会产生一个很大的矩阵，这没办法
        # 100000*50000

        result = (
                x[:, 2 + self.memory_length:-2 - self.memory_length, :] @ torch.transpose(
            self.embedding.weight.to(x.dtype), 0, 1)
        )

        return result, new_memory


class MYTransFormer(nn.Module):

    def __init__(self, myencoder, mydecoder, vocab_size=None, d_model=None):
        super().__init__()

        if vocab_size != None:

            self.embedding = nn.Embedding(vocab_size, d_model).to(model_dtype)

        else:
            self.embedding = None

        self.encoder = myencoder

        self.decoder = mydecoder

    def forward(self, encoder_input, decoder_input, valid_lens=None):
        '''

        :param encoder_input:
        :param decoder_input:
        :param valid_lens: 输入句子的有效长度[batch size , valid_lens]->[batch_size,num_head,valid_lens]
        :return:
        '''

        encoder_input = self.embedding(encoder_input)

        state = self.encoder(encoder_input)

        decoder_input = self.embedding(decoder_input)

        result, state = self.decoder(decoder_input, state, valid_lens)

        result = (
                result @ torch.transpose(self.embedding.weight.to(result.dtype), 0, 1)
        )

        return result, state










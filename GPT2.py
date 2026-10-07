import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GELU

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".local.env")

GPT_CONFIG_BASE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# 不同尺寸的 GPT-2 模型结构差异
MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# ------------------ 模型结构实现 ------------------ #
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attn 子层
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # FFN 子层
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_length = cfg["context_length"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ------------------ 权重加载辅助函数 ------------------ #
def assign(left: torch.Tensor, right: np.ndarray) -> nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配! left: {left.shape}, right: {right.shape}")
    tensor = torch.tensor(right, dtype=left.dtype)
    return nn.Parameter(tensor)


def load_weights_into_gpt(gpt: GPTModel, params: dict) -> None:
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # Transformer blocks
    for b in range(len(params["blocks"])):
        block = gpt.trf_blocks[b]
        p_block = params["blocks"][b]

        # c_attn: 合并的 QKV 权重 & 偏置
        q_w, k_w, v_w = np.split(p_block["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(p_block["attn"]["c_attn"]["b"], 3, axis=-1)

        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        # 注意力输出投影
        block.att.out_proj.weight = assign(
            block.att.out_proj.weight, p_block["attn"]["c_proj"]["w"].T
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias, p_block["attn"]["c_proj"]["b"]
        )

        # MLP
        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight, p_block["mlp"]["c_fc"]["w"].T
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias, p_block["mlp"]["c_fc"]["b"]
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight, p_block["mlp"]["c_proj"]["w"].T
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias, p_block["mlp"]["c_proj"]["b"]
        )

        # LayerNorm
        block.norm1.scale = assign(block.norm1.scale, p_block["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, p_block["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, p_block["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, p_block["ln_2"]["b"])

    # 最后的 LayerNorm & 输出头
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


class LocalGPT2:
    _SIZE_ALIASES = {
        "gpt2-small": "124M",
        "small": "124M",
        "124M": "124M",
        "gpt2-medium": "355M",
        "medium": "355M",
        "355M": "355M",
        "gpt2-large": "774M",
        "large": "774M",
        "774M": "774M",
        "gpt2-xl": "1558M",
        "xl": "1558M",
        "1558M": "1558M",
        "gpt2-small (124M)": "124M",
        "gpt2-medium (355M)": "355M",
        "gpt2-large (774M)": "774M",
        "gpt2-xl (1558M)": "1558M",
    }

    def __init__(
        self,
        model_path: str,
        model_size: str = "124M",
        device: Optional[str] = None,
        use_hf_tokenizer: bool = True,
        tokenizer=None, 
    ):
        """
        model_path: 之前保存好的 .pth 权重文件路径
        model_size: '124M' / '355M' / '774M' / '1558M' 或别名
        tokenizer: 需要有 encode(str) -> list[int], decode(List[int]) -> str
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        size_key = self._normalize_size(model_size)
        if size_key not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 model_size: {model_size}")

        cfg = GPT_CONFIG_BASE.copy()
        cfg.update(MODEL_CONFIGS[size_key])
        cfg["context_length"] = 1024
        cfg["qkv_bias"] = True

        self.cfg = cfg
        self.model = GPTModel(cfg)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.client = OpenAI( api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = tiktoken.get_encoding("gpt2")


    # ---- 静态方法：下载 + 转换 + 保存 .pth ----
    @staticmethod
    def prepare_model_pth(
        model_size: str = "124M",
        save_path: str = "gpt2_124M.pth",
        models_dir: str = "gpt2",
    ) -> str:
        """
        根据传入的模型大小字符串：
          1) 调用 gpt_download.download_and_load_gpt2 下载原始权重
          2) 构建对应结构的 GPTModel
          3) 使用 load_weights_into_gpt 把权重灌进去
          4) 保存为一个 .pth 文件（state_dict）

        返回：保存的 pth 文件路径
        """
        from gpt_download import download_and_load_gpt2  # 延迟导入

        size_key = LocalGPT2._normalize_size(model_size)
        if size_key not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 model_size: {model_size}")

        print(f"[LocalGPT2] 下载并转换 GPT-2 {size_key} 模型权重...")
        settings, params = download_and_load_gpt2(
            model_size=size_key, models_dir=models_dir
        )

        cfg = GPT_CONFIG_BASE.copy()
        cfg.update(MODEL_CONFIGS[size_key])
        cfg["context_length"] = 1024
        cfg["qkv_bias"] = True

        gpt = GPTModel(cfg)
        load_weights_into_gpt(gpt, params)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(gpt.state_dict(), save_path)
        print(f"[LocalGPT2] 模型权重已保存到: {save_path}")
        return save_path

    @staticmethod
    def _normalize_size(model_size: str) -> str:
        key = model_size.strip()
        return LocalGPT2._SIZE_ALIASES.get(key, key)

    # ---- 内部生成函数 ----
    @torch.no_grad()
    def _generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        简单的 top-k 采样生成（单 batch）。
        input_ids: [1, seq_len]
        """
        self.model.eval()
        idx = input_ids.to(self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["context_length"] :]
            logits = self.model(idx_cond)  # [1, T, vocab]
            logits = logits[:, -1, :]      # [1, vocab]

            logits = logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                logits_filtered = torch.full_like(logits, float("-inf"))
                logits_filtered.scatter_(1, ix, v)
                probs = torch.softmax(logits_filtered, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    @staticmethod
    def prepare_dict_and_param_for_generation(pth_path, model_size="124M"):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        
        if not os.path.exists("gpt2"):
            os.makedirs("gpt2")
            
        if not os.path.exists(pth_path):
            LocalGPT2.prepare_model_pth(model_size, pth_path)

    # ---- 对外接口：ask ----
    def ask(
        self,
        question: str,
        max_new_tokens: int = 80,
        top_k: int = 50,
        temperature: float = 1.0,
        only_new_text: bool = True,
        stop: Optional[str] = None,
        deepseek: bool = False,
    ) -> str:
        """
        question: 提问字符串
        ...
        """
        if self.tokenizer is None:
            raise ValueError("当前没有 tokenizer，请在初始化时传入 tokenizer "
                             "或在构造时 use_hf_tokenizer=True")

        # 编码：要求 tokenizer 有 encode / decode
        if deepseek:
            response = self.client.chat.completions.create( model="deepseek-chat", messages=[ {"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}, ], stream=False)
            return response.choices[0].message.content
        
        input_ids_list = self.tokenizer.encode(question)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)

        out_ids = self._generate_ids(
            input_ids,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
        )
        out_ids = out_ids[0].cpu().tolist()

        if only_new_text:
            new_ids = out_ids[len(input_ids_list):]
            text = self.tokenizer.decode(new_ids)
        else:
            text = self.tokenizer.decode(out_ids)

        if stop is not None and stop in text:
            text = text.split(stop, 1)[0]

        return text
    
    



if __name__ == "__main__":
    
    model_size = "124M" # 可选 "124M", "355M", "774M", "1558M"
    # 每个都代表着不同大小的 GPT-2 模型参数
     # "124M" 约 500MB
     # "355M" 约 1.5GB
     # "774M" 约 3GB
     # "1558M" 约 6GB
     
     # 项目路径上不能有中文！！！有报错UTF-8大概率是中文
     # 下载有点慢，而且别把自己电脑撑坏了！
     # 越大的模型回复速度越慢，显存占用也越高。
     
    pth_path = f"checkpoints/gpt2_{model_size}.pth"
    
    # 准备保存目录和模型权重文件
    LocalGPT2.prepare_dict_and_param_for_generation(pth_path)

    bot = LocalGPT2(
        model_path=pth_path,
        model_size=model_size
    )

    ans = bot.ask(
        "Hello? How are you doing today?",
        max_new_tokens=30,
        top_k=50,
        temperature=1.2,
        stop=None,
        deepseek=True
    )
    print("模型回答：", ans)

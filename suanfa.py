"""
ChatGLM3 多模态模型实现
结合 CLIP 视觉编码器和 ChatGLM3 语言模型，实现视觉-语言多模态融合
"""

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel
from peft import LoraConfig, get_peft_model, TaskType
import warnings
from torch.cuda.amp import autocast

# 忽略 PEFT 和 HUGGINGFACE 的一些警告
warnings.filterwarnings("ignore")


# ============================================================================
# 设备与精度配置
# ============================================================================

def get_device():
    """获取可用设备（CUDA 或 CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. 视觉特征 Adapter
# ============================================================================

class VisionAdapter(nn.Module):
    """将 CLIP 特征映射到 LLM 空间并生成可注入的 KV tokens。"""

    def __init__(self, clip_dim: int, llm_dim: int, num_prefix_tokens: int = 32):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens

        self.projection = nn.Linear(clip_dim, llm_dim)
        self.prefix_tokens = nn.Parameter(torch.randn(1, num_prefix_tokens, llm_dim))
        self.adapter_block = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, clip_features: torch.Tensor):
        """
        前向传播
        
        Args:
            clip_features: CLIP 提取的视觉特征 [batch_size, seq_len, clip_dim]
            
        Returns:
            适配后的特征 [batch_size, num_prefix_tokens + seq_len, llm_dim]
        """
        mapped_features = self.projection(clip_features)
        bsz = mapped_features.shape[0]
        adapter_input = torch.cat(
            [self.prefix_tokens.repeat(bsz, 1, 1), mapped_features], dim=1
        )
        return self.adapter_block(adapter_input)


# ============================================================================
# 2. 改造的 ChatGLM3 Attention
# ============================================================================

class ModifiedChatGLMAttention(nn.Module):
    """示意如何在特定层注入视觉 KV。真实环境需改动源码。"""

    def __init__(self, original_attention_module, config, layer_idx, adapter_dim, num_layers=28):
        super().__init__()
        self.original_attention = original_attention_module
        # 在特定层注入视觉信息：10%, 30%, 60%, 90%
        self.inject_layers = set(
            [int(num_layers * 0.1), int(num_layers * 0.3), int(num_layers * 0.6), int(num_layers * 0.9)]
        )
        self.vision_adapter = None
        if layer_idx in self.inject_layers:
            self.vision_adapter = VisionAdapter(clip_dim=adapter_dim, llm_dim=config.hidden_size)

    def forward(self, hidden_states, adapter_output=None, *args, **kwargs):
        """
        前向传播，在特定层注入视觉 KV
        
        Args:
            hidden_states: 隐藏状态
            adapter_output: 视觉适配器输出
        """
        k_vis = v_vis = None
        if self.vision_adapter and adapter_output is not None:
            adapter_kv_tokens = self.vision_adapter(adapter_output)
            k_vis = adapter_kv_tokens
            v_vis = adapter_kv_tokens

        return self.original_attention(
            hidden_states,
            K_vis=k_vis,
            V_vis=v_vis,
            *args,
            **kwargs,
        )


# ============================================================================
# 3. 多模态模型封装
# ============================================================================

class ChatGLM3ForMultimodal(nn.Module):
    """
    ChatGLM3 多模态模型
    结合 CLIP 视觉编码器和 ChatGLM3 语言模型
    """
    
    def __init__(
        self,
        llm_model_path="THUDM/chatglm3-6b",
        clip_model_path="openai/clip-vit-base-patch16",
        torch_dtype=None,
    ):
        super().__init__()
        self.device = get_device()
        self.torch_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        # 冻结的 CLIP 视觉编码器（放在目标 dtype 上，减小显存）
        self.clip = CLIPVisionModel.from_pretrained(clip_model_path, torch_dtype=self.torch_dtype)
        self.clip.requires_grad_(False)
        self.clip = self.clip.to(self.device)
        self.clip_dim = self.clip.config.hidden_size

        # LLM + LoRA
        self.llm = AutoModel.from_pretrained(llm_model_path, trust_remote_code=True, torch_dtype=self.torch_dtype)
        llm_config = self.llm.config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config).to(self.device)

        # 提示：实际替换注意力需修改源码，这里仅示意
        # for i, block in enumerate(self.llm.transformer.layers):
        #     original_attention = block.self_attention
        #     block.self_attention = ModifiedChatGLMAttention(original_attention, llm_config, i, self.clip_dim)

        self.llm.print_trainable_parameters()

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        """
        前向传播
        
        Args:
            images: 输入图像 [batch_size, 3, 224, 224]
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签（可选）[batch_size, seq_len]
            
        Returns:
            LLM 输出（包含 logits 和 loss）
        """
        images = images.to(self.device, dtype=self.torch_dtype)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # 使用 CLIP 提取视觉特征（冻结，不计算梯度）
        with torch.no_grad():
            clip_outputs = self.clip(images)
            visual_features = clip_outputs.last_hidden_state

        # 若使用 GPU，则启用 autocast 以降低显存并加速
        amp_enabled = self.device.type == "cuda"
        with autocast(device_type=self.device.type, enabled=amp_enabled):
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                labels=labels,
            )
        return llm_outputs


# ============================================================================
# 4. 示例运行（轻量化示例，自动选择设备）
# ============================================================================

def example_run():
    """示例运行函数"""
    device = get_device()
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 小 batch / 序列，方便本地快速 smoke test
    bsz, seq_len = 1, 32
    model = ChatGLM3ForMultimodal(torch_dtype=torch_dtype)

    images = torch.randn(bsz, 3, 224, 224, device=device, dtype=torch_dtype)
    input_ids = torch.randint(0, 50000, (bsz, seq_len), device=device)
    attention_mask = torch.ones(bsz, seq_len, device=device)
    labels = input_ids.clone()

    model.train()
    print("\n--- 运行 Forward Pass ---")
    try:
        outputs = model(images, input_ids, attention_mask, labels=labels)
        print(f"logits 形状: {outputs.logits.shape}")
        if outputs.loss is not None:
            print(f"loss: {outputs.loss.item():.4f}")
        print("✅ Forward Pass 成功（假设 ChatGLM3 源码已支持 visual_features）")
    except AttributeError as e:
        print("\n🚨 需要修改 transformers/models/chatglm3/modeling_chatglm.py 以支持 visual_features 参数。")
        print(f"原始错误: {e}")

    print("\n--- 可训练参数统计 ---")
    model.llm.print_trainable_parameters()


if __name__ == "__main__":
    example_run()


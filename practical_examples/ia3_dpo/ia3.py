import torch
import torch.nn as nn
from transformers import LlamaForCausalLM


class IA3LinearLayer(nn.Module):
    """Wraps a linear layer with IA^3 adapter."""
    def __init__(self, module: nn.Linear) -> None:
        super().__init__()
        self.module = module  # pre-trained (frozen) linear layer
        self.adapter = nn.Parameter(torch.ones(1, module.out_features, device=module.weight.device))
        self.use_adapter = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.module(input_tensor)
        if self.use_adapter:
            output = output * self.adapter
        
        return output


def wrap_model(model: LlamaForCausalLM) -> None:
    for _, module in model.model.layers.named_modules():
        # only for Llama-like architectures
        if "LlamaDecoderLayer" in repr(type(module)):
            module.self_attn.k_proj = IA3LinearLayer(module.self_attn.k_proj).to(model.device)
            module.self_attn.v_proj = IA3LinearLayer(module.self_attn.v_proj).to(model.device)
            module.mlp.up_proj = IA3LinearLayer(module.mlp.up_proj).to(model.device) 


def disable_adapters(model: LlamaForCausalLM) -> None:
    for module in model.modules():
        if isinstance(module, IA3LinearLayer):
            module.use_adapter = False


def enable_adapters(model: LlamaForCausalLM) -> None:
    for module in model.modules():
        if isinstance(module, IA3LinearLayer):
            module.use_adapter = True


def load_adapter_weights(model: LlamaForCausalLM, checkpoint_path: str):
    state_dict = torch.load(checkpoint_path)

    for name, module in model.named_modules():
        if name in state_dict:
            module.adapter.data = state_dict[name]["adapter"]

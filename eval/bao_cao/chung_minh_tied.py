"""Chứng minh lỗi: model dùng chung trọng số + modules_to_save (KHÔNG tách) -> gộp/lưu/nạp lại mất lm_head đã train?"""
import sys
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

sys.stdout.reconfigure(encoding="utf-8")
cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True)
m = Qwen2ForCausalLM(cfg)
pc = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                modules_to_save=["embed_tokens", "lm_head"], task_type="CAUSAL_LM")
pm = get_peft_model(m, pc)
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lm_head" in n and "modules_to_save" in n:
            p[-27:] += 0.7
            print("đã 'train' lm_head:", n)
gop = pm.merge_and_unload()
head_da_train = gop.lm_head.weight[-27:].clone()
print("sau gộp: lm_head trùng embed?", gop.lm_head.weight.data_ptr() == gop.get_input_embeddings().weight.data_ptr())
d = tempfile.mkdtemp()
gop.save_pretrained(d)  # giữ nguyên tie_word_embeddings=True (code CŨ)
nap = AutoModelForCausalLM.from_pretrained(d)
con = torch.allclose(nap.lm_head.weight[-27:], head_da_train)
print("CODE CŨ — nạp lại còn lm_head đã train?", con)

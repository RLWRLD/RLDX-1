### Using 🤗 Transformers to Import Model

```python
from modeling_vtc import VTC_Qwen3VL
from transformers import AutoProcessor

model = VTC_Qwen3VL.from_pretrained(
    "RLWRLD/RLDX-1-VLM", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained("RLWRLD/RLDX-1-VLM")
processor.tokenizer.model_max_length = training_args.model_max_length
processor.tokenizer.padding_side = "left"
```
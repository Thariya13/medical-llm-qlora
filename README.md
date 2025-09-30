# 🧠 Medical Reasoning LLM Fine-Tuning with QLoRA 🚀  

Welcome to my **QLoRA fine-tuning project**! 🎯  
In this repo, I explore how to take a strong pretrained LLM and **specialize it in medical reasoning** 🏥📚 using **quantized low-rank adaptation (QLoRA)**.  

The result? A model that **thinks step by step** 🧩 and provides structured answers to medical questions — trained efficiently even on limited GPU resources 💡.  

---

## 📊 Dataset  

We train on **[FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)**, a high-quality dataset built for supervised fine-tuning in medical reasoning tasks.  

- ✅ **Language:** English 🌎  
- ✅ **Samples Used:** `train[:500]` for quick iterations ⚡  
- ✅ **Structure:**  
  - `Question` – user query 🗣️  
  - `Complex_CoT` – detailed chain-of-thought reasoning 💭  
  - `Response` – final concise answer 🎯  

We reformat each entry into **chat-style messages** with a `<think>` tag to explicitly capture reasoning:  

```jsonc
[
  {"role": "user", "content": "Question text"},
  {"role": "assistant", "content": "<think>Reasoning steps</think> Final answer"}
]
```

This design makes the model **simulate structured thinking** before replying 🤓.  

---

## 🏗️ Model & Tokenizer  

For the base LLM, we use:  

- **Model:** [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 🦙  
- **Size:** 1.5B parameters (efficient yet powerful) ⚡  
- **Tokenizer:** The matching tokenizer, customized with a chat template for user/assistant formatting ✨  

This model is distilled, lightweight, and **perfectly suited for QLoRA fine-tuning** 🧑‍💻.  

---

## 🔧 Why QLoRA?  

QLoRA (**Quantized LoRA**) = LoRA adapters + 4-bit quantization.  

Benefits:  
- 🖥️ Train **large models on small GPUs** (memory efficient)  
- 💸 Cut down hardware cost without sacrificing accuracy  
- ⚡ Enable **fast experimentation** with billions of parameters  

We configure QLoRA with:  
- 4-bit quantization (`nf4`)  
- `torch.bfloat16` compute precision  
- `bitsandbytes` for efficient GPU usage  

This combo makes training **accessible and scalable**.  

---

## 🏃‍♂️ Training Pipeline  

- **Libraries Used:**  
  - 🤗 `transformers`  
  - 🤗 `trl` (SFTTrainer)  
  - 🤗 `peft` (LoRA/QLoRA)  
  - ⚡ `bitsandbytes` for quantization  
  - 📊 `datasets`  

- **Steps:**  
  1. Load dataset and reformat to chat-style messages 📝  
  2. Load base LLM in 4-bit quantized mode ⚡  
  3. Attach LoRA adapters via PEFT 🔌  
  4. Fine-tune with `SFTTrainer` 🏋️‍♂️  
  5. Save & merge LoRA weights for inference 💾  

---

## 📈 Inference  

Here’s a quick example for running inference with the fine-tuned QLoRA model:  

```python
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM

# Load the fine-tuned model + tokenizer
model = AutoPeftModelForCausalLM.from_pretrained("output_dir", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("output_dir")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Inference helper
GEN_KW = dict(
    max_new_tokens=512,
    min_new_tokens=32,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)

def test_inference(prompt: str) -> str:
    templated = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = pipe(templated, **GEN_KW)
    return outputs[0]["generated_text"][len(templated):].strip()

# Example
print(test_inference("What is the treatment for hypertension?"))
```

---

## 🎯 Highlights  

✅ **QLoRA fine-tuning** on medical reasoning tasks  
✅ Compact but powerful base model (1.5B params)  
✅ Efficient training with 4-bit quantization  
✅ Structured chain-of-thought reasoning with `<think>` tags  
✅ Easy to reproduce & extend 🚀  

---

## 🤝 Contributing  

Got ideas? PRs welcome! 🙌  
Whether it’s new datasets, training tweaks, or evaluation benchmarks — feel free to collaborate 💡.  

---

## 🌟 Acknowledgements  

- 🤗 Hugging Face for `transformers`, `trl`, `peft`  
- ⚡ `bitsandbytes` for efficient quantization  
- 🏥 FreedomIntelligence for the dataset  
- 🔬 DeepSeek for the distilled base model  

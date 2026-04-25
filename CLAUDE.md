# Narada — Claude Code Notes

## Recurring bugs to never reintroduce

### TRL completions are List[Dict], not strings
TRL >= 0.9 passes `completions` to reward functions as `List[List[Dict]]` (chat format).
Always extract text before passing to any string function:
```python
if isinstance(completion, list):
    text = completion[-1]["content"] if completion else ""
else:
    text = str(completion)
```

### Unsloth telemetry timeout
Unsloth tries to phone home to HF on model load. On Colab this times out after 120s and crashes Cell 3.
Always add before `FastLanguageModel.from_pretrained`:
```python
import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
```

### Qwen3 thinking mode eats all tokens
Qwen3-1.7B generates `<think>...</think>` blocks that consume the entire completion budget,
leaving no room for the JSON action. Always monkey-patch after loading:
```python
_orig = tokenizer.apply_chat_template
def _no_think(*args, **kwargs):
    kwargs["enable_thinking"] = False
    return _orig(*args, **kwargs)
tokenizer.apply_chat_template = _no_think
```

### max_completion_length must be 600, not 300
With 300 tokens, Qwen3 thinking mode + JSON fills the budget and `clipped_ratio` hits 1.0,
giving `reward_std = 0` throughout — the model learns nothing. Always use 600.

### Unsloth compiled cache must be cleared when changing max_completion_length
```python
import shutil, os
if os.path.exists("/content/unsloth_compiled_cache"):
    shutil.rmtree("/content/unsloth_compiled_cache")
```

### nest_asyncio required in Jupyter/Colab
```python
import nest_asyncio
nest_asyncio.apply()
```

## Architecture notes
- Environment server: `src/envs/narada/server/` — changes here require redeploying the HF Space
- Training notebook: `training/narada_grpo.ipynb` — use this for all future runs
- Inference benchmark: `inference.py` — set `GROQ_API_KEY` for Groq, `HF_TOKEN` for HF Router
- `.env` is gitignored — never commit it

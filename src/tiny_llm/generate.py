import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def logsumexp(x, axis=-1):
    c = x.max(axis=axis, keepdims=True)
    return c + mx.log(mx.sum(mx.exp(x - c), axis=axis, keepdims=True))

def softmax(x, axis=-1):
    return mx.exp(x-logsumexp(x, axis=axis))

def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y)
        logits = output_logits[:, -1, :]
        logits = softmax(logits)
        return sampler(logits)
    
    tokenized_prompt = tokenizer._tokenizer.encode(prompt)
    while(True):
        output = _step(model, mx.array([tokenized_prompt])) # B = 1
        token = output.item()
        tokenized_prompt.append(token)
        # print(tokenizer._detokenizer.text)
        tokenizer._detokenizer.add_token(token)
        if(token in tokenizer._eos_token_ids):
            break
        
    tokenizer._detokenizer.finalize()
    print(tokenizer._detokenizer.text)
    return tokenizer._detokenizer.text
    
        


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass

"""Free-text batched sampling; no role protocol or constrained decoding."""
import torch

def generate(model, tokenizer, prompt: torch.Tensor, count: int, cfg, *, row=None):
    """One complete answer per sample. Generation never sees the reference answers."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    samples = []
    try:
        for offset in range(0, count, cfg.generation_batch_size):
            n = min(cfg.generation_batch_size, count - offset)
            ids = prompt.to(device).unsqueeze(0)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    num_return_sequences=n, do_sample=True, temperature=cfg.temperature,
                    top_p=cfg.top_p, top_k=cfg.top_k, max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            for output in outputs:
                tokens = output[prompt.numel():].detach().cpu().clone()
                eos = (tokens == tokenizer.eos_token_id).nonzero()
                if eos.numel():
                    tokens = tokens[:int(eos[0].item()) + 1]
                if tokens.numel() < 1:
                    raise ValueError("Model returned an empty token sequence")
                samples.append((tokens, tokenizer.decode(tokens, skip_special_tokens=True).strip(), None))
    finally:
        model.train(was_training)
    return samples

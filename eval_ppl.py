"""
Perplexity evaluation on WikiText-2 and C4 test sets.

Runs the model forward on test data and computes cross-entropy loss,
returning perplexity = exp(mean_loss).
"""

import torch
import torch.nn.functional as F


def eval_ppl(model, model_name, dataset="wikitext2", seqlen=2048, device="cuda"):
    """Evaluate perplexity on a test dataset.

    Args:
        model: HuggingFace causal LM model.
        model_name: Model name string (for tokenizer).
        dataset: "wikitext2" or "c4".
        seqlen: Sequence length for evaluation chunks.
        device: Device to run evaluation on.

    Returns:
        Perplexity (float).
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    elif dataset == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = "\n\n".join(data[:1100]["text"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc.input_ids[0]  # [total_tokens]
    total_tokens = all_ids.shape[0]

    # Slice into non-overlapping chunks
    nchunks = total_tokens // seqlen
    if nchunks == 0:
        raise ValueError(f"Test text too short: {total_tokens} tokens < {seqlen}")

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(nchunks):
            start = i * seqlen
            input_ids = all_ids[start:start + seqlen].unsqueeze(0).to(device)

            outputs = model(input_ids)
            logits = outputs.logits  # [1, seqlen, vocab_size]

            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_count += shift_labels.numel()

    mean_loss = total_loss / total_count
    ppl = torch.exp(torch.tensor(mean_loss)).item()
    return ppl

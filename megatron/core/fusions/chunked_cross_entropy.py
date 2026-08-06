"""Token-chunked vocab-parallel cross entropy (MetaX memory fix).

Motivation: the stock ``vocab_parallel_cross_entropy`` materializes the full
fp32 logits copy plus the fp32 exp/softmax buffers: for [tokens, vocab] with
vocab=248320, that is tokens x 248320 x 4B x ~2.5 buffers, e.g. ~38 GiB at
mbs=8 (16384 tokens) -- enough to OOM a 64 GiB card on the last PP stage.

This module computes the same loss token-chunk by token-chunk:

* forward (no-grad): per chunk, cast to fp32, logsumexp, subtract target
  logit; chunk intermediates are freed immediately. Only [tokens] fp32 loss
  values are kept.
* backward: per chunk, recompute fp32 softmax (chunk-sized buffer), subtract
  the one-hot target, scale by grad_output, write into a bf16 grad tensor.

Peak extra memory is one chunk of fp32 logits (chunk_size x vocab x 4B),
independent of the total token count. Numerics are bit-identical to the
unfused path when TP=1 (same fp32 math, same order); with TP>1 the logsumexp
uses the standard MAX/SUM all-reduce pair.

Env/arg control lives in language_module.py (CE_TOKEN_CHUNK_SIZE).

20260730 fix: clamp labels at min=0 before gather/scatter_add_. Labels carry
IGNORE_IDX (-100) at masked positions; negative indices are out-of-bounds UB
that CPU bounds-checks loudly but MetaX vendor kernels execute silently with
layout-dependent corruption (observed: nan grad norm at iter1 with
CE_TOKEN_CHUNK_SIZE=512; latent at 2048).
"""

from typing import Optional

import torch
import torch.distributed as dist


class _ChunkedVocabParallelCrossEntropy(torch.autograd.Function):
    """Chunked cross entropy with per-chunk fp32 recompute in backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, chunk_size: int,
                tp_group: Optional[dist.ProcessGroup] = None):
        assert logits.dim() == 3, f"expect [s, b, v] logits, got {logits.shape}"
        ctx.chunk_size = chunk_size
        ctx.tp_group = tp_group
        # chunk over s*b (tokens), NOT just s: with mbs>1, slicing only s leaves
        # the full batch inside one "chunk" (e.g. [2048, 8, 248320] fp32 = 15 GiB).
        s, b, v = logits.shape
        ctx.sb_shape = (s, b)
        flat_logits = logits.reshape(s * b, v)
        flat_labels = labels.reshape(s * b)
        ctx.save_for_backward(flat_logits, flat_labels)

        with torch.no_grad():
            loss_chunks = []
            for i in range(0, s * b, chunk_size):
                lc = flat_logits[i : i + chunk_size].float()  # [cs, v] fp32, freed at loop end
                lse = _tp_logsumexp(lc, dim=-1, tp_group=tp_group)
                # clamp guards IGNORE_IDX (-100) labels: gather with a negative
                # index is out-of-bounds UB (loud on CPU, silent on MetaX vendor
                # kernels). Masked positions are zeroed downstream by loss_mask,
                # so any in-range dummy index is fine here.
                tgt = flat_labels[i : i + chunk_size].long().clamp(min=0)
                pred = lc.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                loss_chunks.append(lse - pred)
            loss = torch.cat(loss_chunks, dim=0)  # [s*b]
        return loss.view(s, b)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_logits, flat_labels = ctx.saved_tensors
        grad = torch.empty_like(flat_logits)  # bf16 [s*b, v]
        n = flat_logits.size(0)
        go = grad_output.reshape(n)
        for i in range(0, n, ctx.chunk_size):
            lc = flat_logits[i : i + ctx.chunk_size].float()  # 唯一的 fp32 chunk buffer
            # clamp guards IGNORE_IDX (-100): scatter_add_ with a negative index
            # is out-of-bounds UB. grad_output is exactly 0 at masked positions
            # (loss = sum(losses * loss_mask)), so the dummy index is a no-op.
            tgt = flat_labels[i : i + ctx.chunk_size].long().clamp(min=0).unsqueeze(-1)
            g = go[i : i + ctx.chunk_size].unsqueeze(-1).to(lc.dtype)
            # in-place: lc = softmax * g - onehot * g  (d(CE)/dlogits * grad_output)
            if ctx.tp_group is not None and ctx.tp_group.size() > 1:
                mx = lc.max(dim=-1, keepdim=True).values
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=ctx.tp_group)
                lc -= mx
            lc.exp_()
            denom = lc.sum(dim=-1, keepdim=True)
            if ctx.tp_group is not None and ctx.tp_group.size() > 1:
                dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            lc *= g / denom
            lc.scatter_add_(-1, tgt, -g)
            grad[i : i + ctx.chunk_size] = lc.to(flat_logits.dtype)
        s, b = ctx.sb_shape
        return grad.view(s, b, -1), None, None, None

def _tp_logsumexp(x: torch.Tensor, dim: int, tp_group) -> torch.Tensor:
    """Logsumexp with optional TP all-reduce (MAX then SUM), fp32."""
    if tp_group is None or tp_group.size() == 1:
        return torch.logsumexp(x, dim=dim)
    mx = x.max(dim=dim, keepdim=True).values
    dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=tp_group)
    sumexp = (x - mx).exp().sum(dim=dim, keepdim=True)
    dist.all_reduce(sumexp, op=dist.ReduceOp.SUM, group=tp_group)
    return (mx + sumexp.log()).squeeze(dim)

"""Core tokenizer and transformer decoder components for MolGen decoding.

This module contains:
- lightweight vocabulary metadata helpers
- SELFIES token ID conversion utilities
- sinusoidal positional encoding
- an embedding-conditioned autoregressive transformer decoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


def _clean_generated_ids(ids: Sequence[int], eos_id: int, pad_id: int) -> List[int]:
    """Drop padding IDs and trim sequence at first EOS token.

    Args:
        ids: Generated token ID sequence.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.

    Returns:
        Clean token IDs suitable for string reconstruction.
    """
    out: List[int] = []
    for token_id in ids:
        # Stop at first EOS because tokens after EOS are not part of the molecule.
        if token_id == eos_id:
            break
        # Skip padding tokens when reconstructing SELFIES.
        if token_id == pad_id:
            continue
        out.append(int(token_id))
    return out


@dataclass
class VocabInfo:
    """Container for token vocabulary and special-token IDs."""

    tokens: List[str]
    token_to_id: Dict[str, int]
    pad_token: str
    bos_token: str
    eos_token: str

    @property
    def pad_id(self) -> int:
        """Return integer ID of pad token."""
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        """Return integer ID of beginning-of-sequence token."""
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        """Return integer ID of end-of-sequence token."""
        return self.token_to_id[self.eos_token]


class SelfiesTokenizer:
    """Tokenizer wrapper around vocab stored in H5 token datasets.

    The training data vocab contains SELFIES symbols and pad token `[nop]`.
    This class appends `<bos>` and `<eos>` if missing so generation can run
    autoregressively.
    """

    PAD = "[nop]"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, base_tokens: Sequence[str]) -> None:
        """Construct tokenizer from base SELFIES token set.

        Args:
            base_tokens: SELFIES vocabulary tokens loaded from dataset.
        """
        tokens = list(base_tokens)
        if self.PAD not in tokens:
            raise ValueError("Expected [nop] in token vocab.")

        # Ensure autoregressive control tokens are always available.
        for special in (self.BOS, self.EOS):
            if special not in tokens:
                tokens.append(special)

        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.info = VocabInfo(
            tokens=tokens,
            token_to_id=token_to_id,
            pad_token=self.PAD,
            bos_token=self.BOS,
            eos_token=self.EOS,
        )

    @property
    def size(self) -> int:
        """Return vocabulary size including special tokens."""
        return len(self.info.tokens)

    @property
    def pad_id(self) -> int:
        """Return pad token ID."""
        return self.info.pad_id

    @property
    def bos_id(self) -> int:
        """Return BOS token ID."""
        return self.info.bos_id

    @property
    def eos_id(self) -> int:
        """Return EOS token ID."""
        return self.info.eos_id

    def ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """Convert integer IDs to token strings.

        Args:
            ids: Sequence of token IDs.

        Returns:
            Token strings in the same order as input IDs.
        """
        return [self.info.tokens[int(i)] for i in ids]

    def ids_to_selfies(self, ids: Sequence[int]) -> str:
        """Convert generated IDs into a SELFIES string.

        Args:
            ids: Sequence of generated token IDs.

        Returns:
            Reconstructed SELFIES string.
        """
        clean_ids = _clean_generated_ids(ids, eos_id=self.eos_id, pad_id=self.pad_id)
        toks = self.ids_to_tokens(clean_ids)
        return "".join(toks)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dynamic max-length growth."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """Precompute positional encodings up to `max_len`.

        Args:
            d_model: Hidden dimension size.
            max_len: Initial maximum sequence length to cache.
        """
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build classic transformer sinusoidal position encoding tensor.

        Args:
            max_len: Number of positions to encode.
            d_model: Hidden dimension size.

        Returns:
            Positional encoding tensor of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(np.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings.

        If a sequence longer than current cache appears, rebuild cache.

        Args:
            x: Token embeddings of shape (B, T, D).

        Returns:
            Position-aware embeddings of shape (B, T, D).
        """
        if x.size(1) > self.pe.size(1):
            new_pe = self._build_pe(x.size(1), self.d_model).to(device=x.device, dtype=x.dtype)
            self.pe = new_pe
        return x + self.pe[:, : x.size(1), :]


class EmbeddingToSelfiesDecoder(nn.Module):
    """Transformer decoder conditioned on a single embedding memory token.

    A molecule embedding is projected to decoder hidden size and treated as
    one memory token. The decoder then autoregressively predicts SELFIES token IDs.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        """Initialize decoder architecture and projection layers.

        Args:
            embedding_dim: Input molecule embedding dimension.
            vocab_size: Decoder vocabulary size.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of decoder layers.
            dim_feedforward: Feed-forward hidden dimension in each layer.
            dropout: Dropout probability.
            max_seq_len: Expected max decoded sequence length.
        """
        super().__init__()
        # Project external molecule embedding to decoder hidden size.
        self.embedding_proj = nn.Linear(embedding_dim, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=max_seq_len + 8)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, mol_emb: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Run one forward pass of teacher-forced decoding.

        Args:
            mol_emb: Molecule embeddings of shape (B, D).
            decoder_input_ids: Decoder token IDs of shape (B, T).

        Returns:
            Vocabulary logits of shape (B, T, V).
        """
        # Condition decoder on one projected memory token per molecule.
        memory = self.embedding_proj(mol_emb).unsqueeze(1)
        tgt = self.token_embed(decoder_input_ids)
        tgt = self.positional(tgt)

        # Causal mask prevents attending to future target positions.
        tgt_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        hidden = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        hidden = self.norm(hidden)
        return self.output(hidden)

    @torch.no_grad()
    def generate(
        self,
        mol_emb: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate token IDs from molecule embeddings.

        Supports greedy decoding or temperature/top-k sampling.

        Args:
            mol_emb: Molecule embeddings of shape (B, D).
            bos_id: Beginning-of-sequence token ID.
            eos_id: End-of-sequence token ID.
            pad_id: Padding token ID.
            max_new_tokens: Maximum generated length (excluding BOS).
            temperature: Softmax temperature for sampling.
            top_k: Truncate sampling distribution to top-k tokens when > 0.
            do_sample: If True, sample; otherwise use greedy argmax decoding.

        Returns:
            Generated token IDs of shape (B, T_gen), excluding BOS.
        """
        self.eval()
        bsz = mol_emb.size(0)
        device = mol_emb.device

        # Seed decoding with BOS token.
        generated = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(mol_emb, generated)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            next_logits = next_logits / temperature

            if do_sample:
                if top_k > 0:
                    # Sample from truncated top-k candidate set.
                    top_vals, top_idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                    probs = torch.softmax(top_vals, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    next_token = top_idx.gather(-1, sampled).squeeze(-1)
                else:
                    # Sample from full distribution.
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding picks the highest-probability next token.
                next_token = torch.argmax(next_logits, dim=-1)

            # Keep finished rows padded so sequence lengths remain aligned.
            next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if torch.all(finished):
                break

        # Drop leading BOS column before returning to callers.
        return generated[:, 1:]

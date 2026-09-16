"""
================================================================
  models.py  —  All 3 models built from scratch
  ┌─────────────────────────────────────────────────────────┐
  │  1. ASRModel        CNN + BiLSTM + CTC                  │
  │  2. TranslatorModel Transformer Encoder-Decoder          │
  │  3. TTSModel        Seq2Seq Attention + Griffin-vocoder  │
  └─────────────────────────────────────────────────────────┘
================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioConfig as AC, ModelConfig as MC

# ═══════════════════════════════════════════════════════════
#  1.  ASR MODEL  —  CNN + BiLSTM + CTC
#
#  Input  : (B, feature_dim, T)   feature_dim = 3 * n_mfcc = 120
#  Output : (B, T, vocab_size)    log-softmax probabilities
# ═══════════════════════════════════════════════════════════
class ASRModel(nn.Module):
    """
    Architecture:
      ┌─────────────────────────────────────────────────────┐
      │  2D CNN  →  extract local spectral-temporal patterns │
      │  BatchNorm + ReLU + MaxPool + Dropout               │
      │  Reshape  →  feed frames into LSTM                  │
      │  4-layer Bidirectional LSTM                         │
      │  Linear  →  vocab projection                        │
      │  Log-Softmax  →  CTC loss compatible                │
      └─────────────────────────────────────────────────────┘
    """

    def __init__(self, feature_dim: int, vocab_size: int):
        super().__init__()
        self.feature_dim = feature_dim   # 120 (MFCC + Δ + ΔΔ)

        # ── CNN block ──────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),     # halve freq axis only
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),     # halve freq axis again
            nn.Dropout2d(0.1),
        )

        # After 2×MaxPool on freq axis:  feature_dim // 4 * 64 channels
        cnn_out = (feature_dim // 4) * 64

        # ── Projection before LSTM ─────────────────────────
        self.cnn_proj = nn.Linear(cnn_out, MC.ASR_HIDDEN)

        # ── BiLSTM ────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = MC.ASR_HIDDEN,
            hidden_size  = MC.ASR_HIDDEN,
            num_layers   = MC.ASR_LSTM_LAYERS,
            batch_first  = True,
            bidirectional= True,
            dropout      = MC.ASR_DROPOUT if MC.ASR_LSTM_LAYERS > 1 else 0,
        )

        # ── Output head ───────────────────────────────────
        self.norm = nn.LayerNorm(MC.ASR_HIDDEN * 2)
        self.fc   = nn.Linear(MC.ASR_HIDDEN * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, feature_dim, T)
        → log-probs : (B, T, vocab_size)
        """
        x = x.unsqueeze(1)              # (B, 1, feature_dim, T)
        x = self.cnn(x)                 # (B, 64, feature_dim//4, T)

        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2)      # (B, T, 64, F)
        x = x.reshape(B, T, C * F)     # (B, T, 64 * F)

        x = self.cnn_proj(x)            # (B, T, hidden)
        x, _ = self.lstm(x)             # (B, T, hidden*2)
        x = self.norm(x)
        x = self.fc(x)                  # (B, T, vocab)
        return x.log_softmax(dim=-1)


# ═══════════════════════════════════════════════════════════
#  2.  TRANSLATOR MODEL  —  Transformer Encoder-Decoder
#
#  Input  : src token ids (B, src_len)
#           tgt token ids (B, tgt_len)    (teacher-forced)
#  Output : logits (B, tgt_len, tgt_vocab_size)
# ═══════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to token embeddings.
    Without this, Transformer has no sense of word order.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TranslatorModel(nn.Module):
    """
    Standard Transformer seq2seq.
    Source and target can have different vocabularies
    (useful for different scripts like Hindi → Tamil).

    Encoder: embeds source tokens → contextual representations
    Decoder: attends over encoder + generates target tokens
    """

    def __init__(self, src_vocab: int, tgt_vocab: int, pad_idx: int = 0):
        super().__init__()
        D = MC.TRANS_D_MODEL
        self.pad_idx = pad_idx

        # ── Embeddings ─────────────────────────────────────
        self.src_emb = nn.Embedding(src_vocab, D, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, D, padding_idx=pad_idx)
        self.src_pe  = PositionalEncoding(D, MC.TRANS_DROPOUT)
        self.tgt_pe  = PositionalEncoding(D, MC.TRANS_DROPOUT)

        # ── Transformer ────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model         = D,
            nhead           = MC.TRANS_NHEAD,
            num_encoder_layers = MC.TRANS_ENC_LAYERS,
            num_decoder_layers = MC.TRANS_DEC_LAYERS,
            dim_feedforward = MC.TRANS_DIM_FF,
            dropout         = MC.TRANS_DROPOUT,
            batch_first     = True,
        )

        # ── Output projection ──────────────────────────────
        self.fc_out = nn.Linear(D, tgt_vocab)

        # Weight tying: share embedding weights with output projection
        # (reduces params, improves generalisation)
        self.fc_out.weight = self.tgt_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """True where token is <pad> (should be ignored)."""
        return x == self.pad_idx

    def _make_causal_mask(self, tgt_len: int, device) -> torch.Tensor:
        """Upper-triangular mask to prevent decoder from seeing future tokens."""
        return torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src : (B, src_len)
        tgt : (B, tgt_len)   — teacher-forced (starts with <sos>)
        → logits : (B, tgt_len, tgt_vocab)
        """
        src_key_pad = self._make_pad_mask(src)
        tgt_key_pad = self._make_pad_mask(tgt)
        tgt_mask    = self._make_causal_mask(tgt.size(1), src.device)

        src_emb = self.src_pe(self.src_emb(src) * math.sqrt(MC.TRANS_D_MODEL))
        tgt_emb = self.tgt_pe(self.tgt_emb(tgt) * math.sqrt(MC.TRANS_D_MODEL))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask           = tgt_mask,
            src_key_padding_mask = src_key_pad,
            tgt_key_padding_mask = tgt_key_pad,
        )                                   # (B, tgt_len, D)
        return self.fc_out(out)             # (B, tgt_len, tgt_vocab)

    @torch.no_grad()
    def translate(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                  max_len: int = 200) -> list[int]:
        """
        Greedy autoregressive decoding at inference time.
        src : (1, src_len)
        """
        self.eval()
        device = src.device

        src_emb  = self.src_pe(self.src_emb(src) * math.sqrt(MC.TRANS_D_MODEL))
        memory   = self.transformer.encoder(src_emb)   # (1, src_len, D)

        tgt_ids  = [sos_idx]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], device=device)
            tgt_emb = self.tgt_pe(self.tgt_emb(tgt) * math.sqrt(MC.TRANS_D_MODEL))
            tgt_mask = self._make_causal_mask(tgt.size(1), device)

            out    = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1, :])       # last token
            next_id = logits.argmax(-1).item()

            if next_id == eos_idx:
                break
            tgt_ids.append(next_id)

        return tgt_ids[1:]   # strip <sos>


# ═══════════════════════════════════════════════════════════
#  3.  TTS MODEL  —  Attention-based Seq2Seq
#
#  Based on simplified Tacotron2 architecture.
#  Input  : text token ids (B, text_len)
#  Output : mel spectrogram (B, n_mels, mel_frames)
# ═══════════════════════════════════════════════════════════
class TTSEncoder(nn.Module):
    """
    Encodes text characters into hidden representations.
    Conv layers capture local phoneme context,
    BiLSTM captures long-range dependencies.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        D = MC.TTS_ENCODER_DIM
        self.embedding = nn.Embedding(vocab_size, D)

        # 3 Conv1D layers (like Tacotron2)
        self.convs = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=5, padding=2),
            nn.BatchNorm1d(D), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(D, D, kernel_size=5, padding=2),
            nn.BatchNorm1d(D), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(D, D, kernel_size=5, padding=2),
            nn.BatchNorm1d(D), nn.ReLU(), nn.Dropout(0.5),
        )
        self.lstm = nn.LSTM(D, D // 2, batch_first=True, bidirectional=True)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text)            # (B, T, D)
        x = self.convs(x.permute(0,2,1))   # (B, D, T)
        x = x.permute(0, 2, 1)             # (B, T, D)
        x, _ = self.lstm(x)                 # (B, T, D)
        return x


class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al.) between decoder state and encoder outputs.
    Tells the decoder WHICH characters to focus on at each mel frame.
    """
    def __init__(self, query_dim: int, key_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_query = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_key   = nn.Linear(key_dim,   attn_dim, bias=False)
        self.v       = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        query : (B, query_dim)
        keys  : (B, src_len, key_dim)
        → context : (B, key_dim),  weights : (B, src_len)
        """
        q = self.W_query(query).unsqueeze(1)   # (B, 1, attn_dim)
        k = self.W_key(keys)                   # (B, src_len, attn_dim)
        e = self.v(torch.tanh(q + k)).squeeze(-1)   # (B, src_len)
        weights  = F.softmax(e, dim=-1)             # (B, src_len)
        context  = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class TTSDecoder(nn.Module):
    """
    Autoregressively generates mel frames one at a time,
    attending over the encoder's character representations.
    """
    def __init__(self):
        super().__init__()
        D       = MC.TTS_ENCODER_DIM
        dec_dim = MC.TTS_DECODER_DIM
        n_mels  = MC.TTS_N_MELS

        self.prenet = nn.Sequential(
            nn.Linear(n_mels, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256),    nn.ReLU(), nn.Dropout(0.5),
        )
        self.attention = BahdanauAttention(dec_dim, D)
        self.lstm1     = nn.LSTMCell(256 + D, dec_dim)
        self.lstm2     = nn.LSTMCell(dec_dim, dec_dim)
        self.mel_proj  = nn.Linear(dec_dim + D, n_mels)
        self.stop_proj = nn.Linear(dec_dim + D, 1)   # stop token prediction

    def forward(self, encoder_out, mel_target=None, teacher_forcing=True):
        """
        encoder_out : (B, text_len, D)
        mel_target  : (B, n_mels, T)   — ground truth mel (for teacher forcing)
        Returns:
            mel_outputs  : (B, n_mels, T)
            stop_outputs : (B, T)
        """
        B, text_len, D = encoder_out.size()
        n_mels  = MC.TTS_N_MELS

        # Initial states
        h1 = torch.zeros(B, MC.TTS_DECODER_DIM, device=encoder_out.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        # Initial mel frame = zeros (silence)
        prev_mel = torch.zeros(B, n_mels, device=encoder_out.device)

        mel_frames  = []
        stop_frames = []

        T = mel_target.size(2) if mel_target is not None else MC.TTS_MAX_FRAMES

        for t in range(T):
            prenet_out = self.prenet(prev_mel)                  # (B, 256)
            context, _ = self.attention(h1, encoder_out)        # (B, D)
            lstm1_in   = torch.cat([prenet_out, context], dim=-1)

            h1, c1  = self.lstm1(lstm1_in, (h1, c1))
            h2, c2  = self.lstm2(h1,       (h2, c2))

            projection_in = torch.cat([h2, context], dim=-1)
            mel_frame  = self.mel_proj(projection_in)           # (B, n_mels)
            stop_token = self.stop_proj(projection_in).squeeze(-1)

            mel_frames.append(mel_frame)
            stop_frames.append(stop_token)

            # Teacher forcing
            if teacher_forcing and mel_target is not None:
                prev_mel = mel_target[:, :, t]
            else:
                prev_mel = mel_frame

        mel_outputs  = torch.stack(mel_frames,  dim=2)   # (B, n_mels, T)
        stop_outputs = torch.stack(stop_frames, dim=1)   # (B, T)
        return mel_outputs, stop_outputs


class TTSModel(nn.Module):
    """
    Full TTS model combining encoder + attention decoder.
    Trained with MSE loss on mel frames + BCE loss on stop tokens.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = TTSEncoder(vocab_size)
        self.decoder = TTSDecoder()

    def forward(self, text, mel_target, teacher_forcing=True):
        encoder_out = self.encoder(text)
        mel_out, stop_out = self.decoder(encoder_out, mel_target, teacher_forcing)
        return mel_out, stop_out

    @torch.no_grad()
    def infer(self, text: torch.Tensor) -> torch.Tensor:
        """
        Generate mel spectrogram for given text at inference.
        text : (1, text_len)
        """
        self.eval()
        enc_out = self.encoder(text)
        mel_out, stop_out = self.decoder(enc_out, mel_target=None, teacher_forcing=False)
        return mel_out   # (1, n_mels, T)

"""
================================================================
  config.py  —  Central configuration for the entire project
================================================================
"""

class AudioConfig:
    SAMPLE_RATE  = 16_000
    N_MFCC       = 40
    N_FFT        = 512
    HOP_LENGTH   = 160       # 10ms at 16kHz
    WIN_LENGTH   = 400       # 25ms at 16kHz
    N_MELS       = 80
    MAX_AUDIO_LEN = 15       # seconds — skip longer clips

class ModelConfig:
    # ── ASR ──────────────────────────────────────────────
    ASR_HIDDEN      = 512
    ASR_LSTM_LAYERS = 4
    ASR_DROPOUT     = 0.3

    # ── Transformer (Translation) ─────────────────────────
    TRANS_D_MODEL   = 256
    TRANS_NHEAD     = 8
    TRANS_ENC_LAYERS= 4
    TRANS_DEC_LAYERS= 4
    TRANS_DIM_FF    = 1024
    TRANS_DROPOUT   = 0.1
    TRANS_MAX_LEN   = 256    # max token sequence length

    # ── TTS ──────────────────────────────────────────────
    TTS_ENCODER_DIM = 256
    TTS_DECODER_DIM = 512
    TTS_N_MELS      = 80
    TTS_MAX_FRAMES  = 1000

class TrainConfig:
    BATCH_SIZE     = 16
    EPOCHS         = 50
    LR             = 1e-3
    WARMUP_STEPS   = 4000
    GRAD_CLIP      = 1.0
    PATIENCE       = 7
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR        = "logs"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

================================================================
  SPEECH-TO-SPEECH TRANSLATION  —  FROM SCRATCH
================================================================

PROJECT STRUCTURE
─────────────────
  config.py      ← all hyperparameters in one place
  tokenizer.py   ← character-level tokenizer (Hindi, Tamil, English, etc.)
  dataset.py     ← data loaders for ASR, Translation, TTS
  models.py      ← all 3 models built from scratch
  trainer.py     ← training loops with early stopping + checkpointing
  pipeline.py    ← end-to-end inference + Griffin-Lim vocoder
  requirements.txt

INSTALL
───────
  pip install -r requirements.txt

DATA TO DOWNLOAD
────────────────
  ASR (Hindi)       →  automatic via HuggingFace:
                        datasets.load("ai4bharat/kathbath", "hindi")

  Translation       →  automatic via HuggingFace:
                        datasets.load("ai4bharat/samanantar", "hi")

  TTS (English)     →  manual download (free):
                        wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
                        tar -xf LJSpeech-1.1.tar.bz2 -C data/tts/english/

  TTS (Hindi)       →  manual download from IIT Madras:
                        https://www.iitm.ac.in/donlab/tts/database.php

HOW TO RUN
──────────
  # Train everything + run inference:
  python pipeline.py

  # Or train each stage separately:
  from trainer import train_asr, train_translation, train_tts

LANGUAGE PAIRS SUPPORTED
─────────────────────────
  The system supports any Indian language pair.
  Change these two variables in pipeline.py:
    lang      = "hindi"      ← ASR input language
    src_lang  = "hi"         ← Translation source
    tgt_lang  = "en"         ← Translation target

  Available languages in Kathbath:
    hindi, tamil, telugu, kannada, bengali,
    marathi, gujarati, odia, punjabi, assamese, maithili, urdu

  Available language codes for Samanantar:
    hi, ta, te, kn, bn, mr, gu, or, pa, as, mai, ur

ARCHITECTURE SUMMARY
─────────────────────
  ┌────────────────────────────────────────────────────────┐
  │  Stage 1 — ASR                                         │
  │    Input  : Raw audio (16kHz .wav)                     │
  │    Feature: MFCC + Delta + Delta-Delta  (120 features) │
  │    Model  : 2×CNN + 4×BiLSTM + CTC Loss                │
  │    Output : Transcribed text (Hindi)                   │
  │    Metric : Word Error Rate (WER)                      │
  ├────────────────────────────────────────────────────────┤
  │  Stage 2 — Translation                                 │
  │    Input  : Source text (Hindi)                        │
  │    Model  : Transformer Encoder-Decoder                │
  │             4 enc layers + 4 dec layers                │
  │             Bahdanau attention + label smoothing       │
  │             Warmup LR scheduler                        │
  │    Output : Target text (English)                      │
  │    Metric : BLEU Score                                 │
  ├────────────────────────────────────────────────────────┤
  │  Stage 3 — TTS                                         │
  │    Input  : Target text (English)                      │
  │    Model  : Conv Encoder + Bahdanau Attention          │
  │             + 2-layer LSTM Decoder (Tacotron2-style)   │
  │    Output : Log-Mel spectrogram                        │
  │    Vocoder: Griffin-Lim (classical, no training)       │
  │    Metric : MSE on mel + BCE on stop tokens            │
  └────────────────────────────────────────────────────────┘

EXPECTED TRAINING TIME (GPU: RTX 3080 / A100)
──────────────────────────────────────────────
  ASR          ~8 hours   (Kathbath hindi, 50 epochs)
  Translation  ~3 hours   (Samanantar 200k pairs, 50 epochs)
  TTS          ~8 hours   (LJSpeech 13k samples, 50 epochs)

EXPECTED RESULTS AFTER TRAINING
─────────────────────────────────
  ASR WER      :  ~25-35%  (train-clean, Hindi)
  BLEU Score   :  ~15-25   (Hindi → English)
  TTS          :  Intelligible speech output

================================================================

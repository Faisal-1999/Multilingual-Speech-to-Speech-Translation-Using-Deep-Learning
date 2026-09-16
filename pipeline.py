"""
================================================================
  pipeline.py  —  Full speech-to-speech inference pipeline
                  + Griffin-Lim vocoder (mel → audio)
================================================================
  Flow:
    Input audio (Hindi)
      → ASR        → Hindi text
      → Translator → English text   (or any target lang)
      → TTS        → Mel spectrogram
      → Vocoder    → Output audio (English)
================================================================
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import scipy.io.wavfile as wav

from config import AudioConfig as AC, DEVICE
from tokenizer import CharTokenizer
from models   import ASRModel, TranslatorModel, TTSModel
from dataset  import FeatureExtractor


# ─────────────────────────────────────────────────────────────
#  GRIFFIN-LIM VOCODER  (convert mel → audio without neural net)
#  This is a classical phase-reconstruction algorithm.
#  For production use, replace with HiFi-GAN vocoder.
# ─────────────────────────────────────────────────────────────
class GriffinLimVocoder:
    """
    Converts log-mel spectrogram → audio waveform using
    the Griffin-Lim iterative phase reconstruction algorithm.

    Advantages  : No training needed, simple, CPU-friendly
    Disadvantage: Audio quality is lower than neural vocoders
    """

    def __init__(self, n_iter: int = 60):
        self.n_iter = n_iter
        self.griffin_lim = T.GriffinLim(
            n_fft      = AC.N_FFT,
            win_length = AC.WIN_LENGTH,
            hop_length = AC.HOP_LENGTH,
            n_iter     = n_iter,
        )
        self.inv_mel = T.InverseMelScale(
            n_stft    = AC.N_FFT // 2 + 1,
            n_mels    = AC.N_MELS,
            sample_rate = AC.SAMPLE_RATE,
        )

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        log_mel : (n_mels, T)
        → waveform : (T_audio,)
        """
        mel    = torch.exp(log_mel)              # undo log
        linear = self.inv_mel(mel)               # mel → linear spectrogram
        audio  = self.griffin_lim(linear)        # spectrogram → waveform
        return audio

    def save(self, audio: torch.Tensor, path: str):
        audio_np = audio.numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int = (audio_np * 32767).astype(np.int16)
        wav.write(path, AC.SAMPLE_RATE, audio_int)
        print(f"  🔊  Audio saved → {path}")


# ─────────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────────
class SpeechToSpeechPipeline:
    """
    Combines all 3 models into a single end-to-end pipeline.

    Usage:
        pipeline = SpeechToSpeechPipeline(
            asr_ckpt   = "checkpoints/asr_best.pt",
            trans_ckpt = "checkpoints/translator_best.pt",
            tts_ckpt   = "checkpoints/tts_best.pt",
            src_tokenizer = hindi_tokenizer,
            tgt_tokenizer = english_tokenizer,
        )
        pipeline.translate("input_hindi.wav", "output_english.wav")
    """

    def __init__(self,
                 asr_ckpt:   str,
                 trans_ckpt: str,
                 tts_ckpt:   str,
                 src_tokenizer: CharTokenizer,
                 tgt_tokenizer: CharTokenizer,
                 feature_dim: int = 120):    # 3 * 40 MFCC

        print("[Pipeline] Loading models ...")
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.extractor     = FeatureExtractor(augment=False)
        self.vocoder       = GriffinLimVocoder()

        # ── Load ASR ─────────────────────────────────────
        self.asr = ASRModel(feature_dim, src_tokenizer.vocab_size).to(DEVICE)
        self.asr.load_state_dict(
            torch.load(asr_ckpt, map_location=DEVICE)["model_state"]
        )
        self.asr.eval()
        print("  ✅  ASR model loaded")

        # ── Load Translator ───────────────────────────────
        self.translator = TranslatorModel(
            src_vocab = src_tokenizer.vocab_size,
            tgt_vocab = tgt_tokenizer.vocab_size,
        ).to(DEVICE)
        self.translator.load_state_dict(
            torch.load(trans_ckpt, map_location=DEVICE)["model_state"]
        )
        self.translator.eval()
        print("  ✅  Translator model loaded")

        # ── Load TTS ─────────────────────────────────────
        self.tts = TTSModel(tgt_tokenizer.vocab_size).to(DEVICE)
        self.tts.load_state_dict(
            torch.load(tts_ckpt, map_location=DEVICE)["model_state"]
        )
        self.tts.eval()
        print("  ✅  TTS model loaded")
        print("[Pipeline] Ready.\n")

    def transcribe(self, audio_path: str) -> str:
        """Step 1: Audio → source language text."""
        wav, sr = torchaudio.load(audio_path)
        if sr != AC.SAMPLE_RATE:
            wav = T.Resample(sr, AC.SAMPLE_RATE)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        features = self.extractor(wav).unsqueeze(0).to(DEVICE)  # (1, 120, T)
        with torch.no_grad():
            log_probs = self.asr(features)           # (1, T, vocab)
        pred_ids  = log_probs.argmax(-1).squeeze().tolist()
        return self.src_tokenizer.ctc_decode(pred_ids)

    def translate_text(self, src_text: str) -> str:
        """Step 2: Source text → target language text."""
        src_ids = torch.tensor(
            [self.src_tokenizer.encode(src_text)], dtype=torch.long
        ).to(DEVICE)
        pred_ids = self.translator.translate(
            src_ids,
            sos_idx = self.tgt_tokenizer.SOS_IDX,
            eos_idx = self.tgt_tokenizer.EOS_IDX,
        )
        return self.tgt_tokenizer.decode(pred_ids)

    def synthesize(self, text: str, output_path: str):
        """Step 3: Target language text → audio."""
        text_ids = torch.tensor(
            [self.tgt_tokenizer.encode(text)], dtype=torch.long
        ).to(DEVICE)
        with torch.no_grad():
            mel = self.tts.infer(text_ids).squeeze(0).cpu()  # (n_mels, T)
        audio = self.vocoder(mel)
        self.vocoder.save(audio, output_path)

    def translate(self, input_audio: str, output_audio: str) -> dict:
        """
        Full end-to-end translation.

        Returns a dict with intermediate results for inspection.
        """
        print(f"[Pipeline] Input  : {input_audio}")

        # Step 1
        src_text = self.transcribe(input_audio)
        print(f"[Pipeline] ASR    : {src_text}")

        # Step 2
        tgt_text = self.translate_text(src_text)
        print(f"[Pipeline] Transl : {tgt_text}")

        # Step 3
        self.synthesize(tgt_text, output_audio)
        print(f"[Pipeline] Output : {output_audio}")

        return {
            "input_audio" : input_audio,
            "src_text"    : src_text,
            "tgt_text"    : tgt_text,
            "output_audio": output_audio,
        }


# ─────────────────────────────────────────────────────────────
#  MAIN — train all 3 models then run inference
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader, random_split
    from dataset import (
        ASRDataset,    asr_collate,
        TranslationDataset, translation_collate,
        TTSDataset,    tts_collate,
    )
    from trainer import train_asr, train_translation, train_tts

    # ── Tokenizers ────────────────────────────────────────
    src_tok = CharTokenizer()   # Hindi  (source)
    tgt_tok = CharTokenizer()   # English (target)
    tts_tok = CharTokenizer()   # English (for TTS — same as tgt_tok)

    # ─────────────────────────────────────────────────────
    #  STEP A  :  Train ASR  (Hindi speech → Hindi text)
    # ─────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  STAGE 1 : ASR  (Hindi speech → Hindi text)")
    print("━"*60)

    asr_dataset = ASRDataset.from_huggingface(
        "ai4bharat/kathbath",
        tokenizer = src_tok,
        lang      = "hindi",
        split     = "train",
        augment   = True,
    )
    n_val  = int(len(asr_dataset) * 0.1)
    n_train = len(asr_dataset) - n_val
    asr_train, asr_val = random_split(asr_dataset, [n_train, n_val])

    asr_train_loader = DataLoader(asr_train, batch_size=16, shuffle=True,
                                  collate_fn=asr_collate, num_workers=4)
    asr_val_loader   = DataLoader(asr_val,   batch_size=16, shuffle=False,
                                  collate_fn=asr_collate, num_workers=2)

    feature_dim = 3 * 40   # MFCC + Δ + ΔΔ
    asr_model   = ASRModel(feature_dim, src_tok.vocab_size)
    train_asr(asr_model, asr_train_loader, asr_val_loader, src_tok)
    src_tok.save("checkpoints/src_tokenizer.json")

    # ─────────────────────────────────────────────────────
    #  STEP B  :  Train Translator  (Hindi text → English text)
    # ─────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  STAGE 2 : Translation  (Hindi text → English text)")
    print("━"*60)

    trans_dataset = TranslationDataset.from_huggingface(
        src_lang      = "hi",
        tgt_lang      = "en",
        src_tokenizer = src_tok,
        tgt_tokenizer = tgt_tok,
        max_samples   = 200_000,
    )
    n_val   = int(len(trans_dataset) * 0.05)
    n_train = len(trans_dataset) - n_val
    tr_train, tr_val = random_split(trans_dataset, [n_train, n_val])

    tr_train_loader = DataLoader(tr_train, batch_size=64, shuffle=True,
                                 collate_fn=translation_collate, num_workers=4)
    tr_val_loader   = DataLoader(tr_val,   batch_size=64, shuffle=False,
                                 collate_fn=translation_collate, num_workers=2)

    trans_model = TranslatorModel(src_tok.vocab_size, tgt_tok.vocab_size)
    train_translation(trans_model, tr_train_loader, tr_val_loader, tgt_tok)
    tgt_tok.save("checkpoints/tgt_tokenizer.json")

    # ─────────────────────────────────────────────────────
    #  STEP C  :  Train TTS  (English text → English speech)
    # ─────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  STAGE 3 : TTS  (English text → audio)")
    print("━"*60)

    # Download LJSpeech (English TTS dataset):
    #   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    #   tar -xf LJSpeech-1.1.tar.bz2 -C data/tts/english/

    tts_dataset = TTSDataset.from_folder("data/tts/english/LJSpeech-1.1/", tts_tok)
    n_val   = int(len(tts_dataset) * 0.05)
    n_train = len(tts_dataset) - n_val
    tts_train, tts_val = random_split(tts_dataset, [n_train, n_val])

    tts_train_loader = DataLoader(tts_train, batch_size=16, shuffle=True,
                                  collate_fn=tts_collate, num_workers=4)
    tts_val_loader   = DataLoader(tts_val,   batch_size=16, shuffle=False,
                                  collate_fn=tts_collate, num_workers=2)

    tts_model = TTSModel(tts_tok.vocab_size)
    train_tts(tts_model, tts_train_loader, tts_val_loader)
    tts_tok.save("checkpoints/tts_tokenizer.json")

    # ─────────────────────────────────────────────────────
    #  STEP D  :  Run Full Pipeline
    # ─────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  STAGE 4 : End-to-End Inference")
    print("━"*60)

    # Reload tokenizers (in case vocab was updated)
    src_tok.load("checkpoints/src_tokenizer.json")
    tgt_tok.load("checkpoints/tgt_tokenizer.json")
    tts_tok.load("checkpoints/tts_tokenizer.json")

    pipeline = SpeechToSpeechPipeline(
        asr_ckpt   = "checkpoints/asr_best.pt",
        trans_ckpt = "checkpoints/translator_best.pt",
        tts_ckpt   = "checkpoints/tts_best.pt",
        src_tokenizer = src_tok,
        tgt_tokenizer = tgt_tok,
        feature_dim   = 120,
    )

    result = pipeline.translate(
        input_audio  = "test_hindi.wav",     # your Hindi input audio
        output_audio = "output_english.wav",
    )

    print("\n  ✅  Done!")
    print(f"  Source text      : {result['src_text']}")
    print(f"  Translated text  : {result['tgt_text']}")
    print(f"  Output saved to  : {result['output_audio']}")

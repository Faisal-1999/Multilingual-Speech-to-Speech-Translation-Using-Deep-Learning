"""
================================================================
  dataset.py  —  Dataset classes for all 3 stages
                 Works with Kathbath (Indian languages)
                 and LibriSpeech (English)
================================================================

Supported datasets (all free):
  ASR   → Kathbath   : huggingface.co/datasets/ai4bharat/kathbath
          LibriSpeech : huggingface.co/datasets/librispeech_asr
  Trans → Samanantar  : huggingface.co/datasets/ai4bharat/samanantar
  TTS   → IndicTTS   : download from iitm.ac.in/donlab/tts/
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchaudio
import torchaudio.transforms as T
import numpy as np

from config import AudioConfig as AC, TrainConfig as TC
from tokenizer import CharTokenizer

# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTOR  (shared by ASR + TTS datasets)
# ─────────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    Converts raw waveform → MFCC + Delta + Delta-Delta (3×n_mfcc features).

    Why deltas?
      • MFCC    : static spectral snapshot
      • Delta   : velocity  (first-order temporal change)
      • Delta-2 : acceleration (second-order temporal change)
    Together they give the model a sense of HOW speech is changing,
    not just what it sounds like at a moment.
    """

    def __init__(self, augment: bool = False):
        self.augment = augment
        self.mfcc_transform = T.MFCC(
            sample_rate = AC.SAMPLE_RATE,
            n_mfcc      = AC.N_MFCC,
            melkwargs   = {
                "n_fft"      : AC.N_FFT,
                "hop_length" : AC.HOP_LENGTH,
                "win_length" : AC.WIN_LENGTH,
                "n_mels"     : AC.N_MELS,
            },
        )
        # SpecAugment for training robustness
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : (1, T)
        returns  : (3 * n_mfcc, time_frames)  — MFCC + Δ + ΔΔ
        """
        mfcc  = self.mfcc_transform(waveform).squeeze(0)  # (n_mfcc, T)

        # Compute deltas
        delta  = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)

        features = torch.cat([mfcc, delta, delta2], dim=0)  # (3*n_mfcc, T)

        # Per-utterance normalisation
        features = (features - features.mean()) / (features.std() + 1e-8)

        # SpecAugment (training only)
        if self.augment:
            features = self.freq_mask(features)
            features = self.time_mask(features)

        return features   # (120, T) when n_mfcc=40


# ─────────────────────────────────────────────────────────────
#  1.  ASR DATASET
# ─────────────────────────────────────────────────────────────
class ASRDataset(Dataset):
    """
    Loads speech + transcript pairs.

    Supports two backends:
      • HuggingFace datasets (Kathbath, LibriSpeech)
      • Local folder: each line in manifest.txt = "audio_path|transcript"

    Usage:
        # HuggingFace:
        ds = ASRDataset.from_huggingface("ai4bharat/kathbath", lang="hindi", split="train")

        # Local files:
        ds = ASRDataset.from_manifest("data/hindi/train_manifest.txt")
    """

    def __init__(self, samples: list[dict], tokenizer: CharTokenizer,
                 augment: bool = False):
        """
        samples: list of {"audio_path": str, "transcript": str}
                 OR {"waveform": tensor, "sr": int, "transcript": str}
        """
        self.samples   = samples
        self.tokenizer = tokenizer
        self.extractor = FeatureExtractor(augment=augment)
        self.max_samples = int(AC.MAX_AUDIO_LEN * AC.SAMPLE_RATE)

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def from_huggingface(cls, dataset_name: str, tokenizer: CharTokenizer,
                         lang: str = "hindi", split: str = "train",
                         augment: bool = False):
        """
        Load from HuggingFace hub.
        dataset_name examples:
          "ai4bharat/kathbath"   — 12 Indian languages
          "librispeech_asr"      — English
        """
        from datasets import load_dataset
        print(f"[ASRDataset] Loading {dataset_name} | lang={lang} | split={split}")

        if "kathbath" in dataset_name:
            raw = load_dataset(dataset_name, lang, split=split,
                               trust_remote_code=True)
            samples = [
                {"waveform": torch.tensor(s["audio"]["array"], dtype=torch.float32),
                 "sr"       : s["audio"]["sampling_rate"],
                 "transcript": s["transcript"]}
                for s in raw
            ]
        elif "librispeech" in dataset_name:
            raw = load_dataset(dataset_name, "clean", split=split,
                               trust_remote_code=True)
            samples = [
                {"waveform": torch.tensor(s["audio"]["array"], dtype=torch.float32),
                 "sr"       : s["audio"]["sampling_rate"],
                 "transcript": s["text"]}
                for s in raw
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Build vocab from all transcripts
        all_texts = [s["transcript"] for s in samples]
        tokenizer.build_vocab(all_texts)
        print(f"[ASRDataset] {len(samples)} samples loaded.")
        return cls(samples, tokenizer, augment)

    @classmethod
    def from_manifest(cls, manifest_path: str, tokenizer: CharTokenizer,
                      augment: bool = False):
        """
        Load from a local manifest file.
        Each line: /path/to/audio.wav|transcript text here
        """
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    samples.append({"audio_path": parts[0], "transcript": parts[1]})
        all_texts = [s["transcript"] for s in samples]
        tokenizer.build_vocab(all_texts)
        print(f"[ASRDataset] {len(samples)} samples from {manifest_path}")
        return cls(samples, tokenizer, augment)

    # ── Core Dataset Methods ──────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, sample: dict) -> torch.Tensor:
        if "waveform" in sample:
            wav = sample["waveform"].unsqueeze(0)   # (1, T)
            sr  = sample["sr"]
        else:
            wav, sr = torchaudio.load(sample["audio_path"])

        # Resample if needed
        if sr != AC.SAMPLE_RATE:
            wav = T.Resample(sr, AC.SAMPLE_RATE)(wav)

        # Convert stereo to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Truncate very long clips
        wav = wav[:, :self.max_samples]

        return wav   # (1, T)

    def __getitem__(self, idx):
        sample     = self.samples[idx]
        waveform   = self._load_waveform(sample)
        features   = self.extractor(waveform)       # (120, T)
        label      = torch.tensor(
            self.tokenizer.encode(sample["transcript"]),
            dtype=torch.long
        )
        return features, label


# ── ASR Collate ───────────────────────────────────────────────
def asr_collate(batch):
    """
    Pads features + labels for a batch.
    Returns what nn.CTCLoss expects.
    """
    features, labels = zip(*batch)

    input_lengths  = torch.tensor([f.shape[1] for f in features], dtype=torch.long)
    label_lengths  = torch.tensor([len(l)     for l in labels],   dtype=torch.long)

    # Pad features → (B, feature_dim, T_max)
    features_padded = pad_sequence(
        [f.T for f in features], batch_first=True, padding_value=0.0
    ).permute(0, 2, 1)

    labels_concat = torch.cat(labels)   # CTCLoss takes flat label tensor

    return features_padded, labels_concat, input_lengths, label_lengths


# ─────────────────────────────────────────────────────────────
#  2.  TRANSLATION DATASET
# ─────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    """
    Pairs of (source text, target text) for seq2seq training.

    Supported sources:
      • HuggingFace Samanantar (11 Indian langs ↔ English)
      • Local TSV:  source_text\ttarget_text  per line

    Usage:
        ds = TranslationDataset.from_huggingface(
            src_lang="hi", tgt_lang="en",
            src_tokenizer=hi_tok, tgt_tokenizer=en_tok
        )
    """

    def __init__(self, pairs: list[tuple[str, str]],
                 src_tokenizer: CharTokenizer,
                 tgt_tokenizer: CharTokenizer,
                 max_len: int = 256):
        self.pairs         = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len       = max_len

    @classmethod
    def from_huggingface(cls, src_lang: str, tgt_lang: str,
                         src_tokenizer: CharTokenizer,
                         tgt_tokenizer: CharTokenizer,
                         split: str = "train",
                         max_samples: int = 100_000):
        """
        Load from Samanantar dataset.
        src_lang / tgt_lang : ISO 639-1 codes  e.g. "hi", "ta", "en"
        """
        from datasets import load_dataset

        # Samanantar always pairs Indian language ↔ English
        indic_lang = src_lang if src_lang != "en" else tgt_lang
        print(f"[TranslationDataset] Loading Samanantar {indic_lang}↔en ...")

        raw = load_dataset("ai4bharat/samanantar", indic_lang, split=split,
                           trust_remote_code=True)
        raw = raw.select(range(min(max_samples, len(raw))))

        if src_lang == "en":
            pairs = [(s["src"], s["tgt"]) for s in raw]   # en → indic
        else:
            pairs = [(s["tgt"], s["src"]) for s in raw]   # indic → en

        src_texts = [p[0] for p in pairs]
        tgt_texts = [p[1] for p in pairs]
        src_tokenizer.build_vocab(src_texts)
        tgt_tokenizer.build_vocab(tgt_texts)
        print(f"[TranslationDataset] {len(pairs)} pairs loaded.")
        return cls(pairs, src_tokenizer, tgt_tokenizer)

    @classmethod
    def from_tsv(cls, path: str,
                 src_tokenizer: CharTokenizer,
                 tgt_tokenizer: CharTokenizer):
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        src_tokenizer.build_vocab([p[0] for p in pairs])
        tgt_tokenizer.build_vocab([p[1] for p in pairs])
        return cls(pairs, src_tokenizer, tgt_tokenizer)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        src_ids = self.src_tokenizer.encode(src_text)[:self.max_len]
        # Teacher-forced target: decoder input has <sos>, output has <eos>
        tgt_in  = self.tgt_tokenizer.encode(tgt_text, add_sos=True)[:self.max_len]
        tgt_out = self.tgt_tokenizer.encode(tgt_text, add_eos=True)[:self.max_len]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_in,  dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def translation_collate(batch):
    src, tgt_in, tgt_out = zip(*batch)
    src_pad     = pad_sequence(src,     batch_first=True, padding_value=CharTokenizer.PAD_IDX)
    tgt_in_pad  = pad_sequence(tgt_in,  batch_first=True, padding_value=CharTokenizer.PAD_IDX)
    tgt_out_pad = pad_sequence(tgt_out, batch_first=True, padding_value=CharTokenizer.PAD_IDX)
    return src_pad, tgt_in_pad, tgt_out_pad


# ─────────────────────────────────────────────────────────────
#  3.  TTS DATASET
# ─────────────────────────────────────────────────────────────
class TTSDataset(Dataset):
    """
    (text, mel_spectrogram) pairs for TTS training.

    Dataset options:
      • IndicTTS  : iitm.ac.in/donlab/tts/  (Hindi, Tamil, Telugu, etc.)
      • LJSpeech  : keithito.com/lj-speech-dataset/  (English)

    Local folder structure expected:
        data/tts/hindi/
            wavs/            ← .wav files
            metadata.csv     ← filename|normalized_text

    Usage:
        ds = TTSDataset.from_folder("data/tts/hindi/", tokenizer)
    """

    def __init__(self, samples: list[dict], tokenizer: CharTokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.mel_transform = T.MelSpectrogram(
            sample_rate = AC.SAMPLE_RATE,
            n_fft       = AC.N_FFT,
            hop_length  = AC.HOP_LENGTH,
            win_length  = AC.WIN_LENGTH,
            n_mels      = AC.N_MELS,
        )

    @classmethod
    def from_folder(cls, folder: str, tokenizer: CharTokenizer):
        """
        Reads metadata.csv (LJSpeech/IndicTTS format):
        filename|raw_text|normalized_text  (uses normalized_text)
        """
        import os
        metadata_path = os.path.join(folder, "metadata.csv")
        wav_dir       = os.path.join(folder, "wavs")
        samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    fname    = parts[0]
                    text     = parts[-1]   # normalized text is last column
                    wav_path = os.path.join(wav_dir, fname + ".wav")
                    if os.path.exists(wav_path):
                        samples.append({"audio_path": wav_path, "text": text})
        tokenizer.build_vocab([s["text"] for s in samples])
        print(f"[TTSDataset] {len(samples)} samples from {folder}")
        return cls(samples, tokenizer)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        wav, sr  = torchaudio.load(sample["audio_path"])
        if sr != AC.SAMPLE_RATE:
            wav = T.Resample(sr, AC.SAMPLE_RATE)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        mel = self.mel_transform(wav).squeeze(0)          # (n_mels, T)
        mel = torch.log(mel + 1e-9)                       # log mel
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)    # normalise

        text_ids = torch.tensor(
            self.tokenizer.encode(sample["text"]), dtype=torch.long
        )
        return text_ids, mel


def tts_collate(batch):
    texts, mels = zip(*batch)
    text_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    mel_lengths  = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    texts_padded = pad_sequence(texts, batch_first=True,
                                padding_value=CharTokenizer.PAD_IDX)
    # Pad mel along time axis
    max_mel_len  = max(m.shape[1] for m in mels)
    mels_padded  = torch.zeros(len(mels), AC.N_MELS, max_mel_len)
    for i, m in enumerate(mels):
        mels_padded[i, :, :m.shape[1]] = m
    return texts_padded, mels_padded, text_lengths, mel_lengths

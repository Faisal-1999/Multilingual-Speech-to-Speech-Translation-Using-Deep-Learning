"""
================================================================
  trainer.py  —  Training loops for ASR, Translation, TTS
================================================================
"""

import os, csv, time, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from jiwer import wer as compute_wer
from sacrebleu.metrics import BLEU

from config import TrainConfig as TC, DEVICE
from models import ASRModel, TranslatorModel, TTSModel
from tokenizer import CharTokenizer

os.makedirs(TC.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TC.LOG_DIR,        exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  WARMUP SCHEDULER  (standard for Transformers)
# ─────────────────────────────────────────────────────────────
class WarmupScheduler:
    """
    Linear warmup then inverse-sqrt decay.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self):
        self._step += 1
        lr = (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * self.warmup_steps ** -1.5
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, loss, name):
    path = os.path.join(TC.CHECKPOINT_DIR, f"{name}_best.pt")
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss"       : loss,
    }, path)
    print(f"  ✅  Saved best {name} checkpoint  →  {path}")


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  📦  Loaded checkpoint from {path}")
    return model


# ─────────────────────────────────────────────────────────────
#  1.  ASR TRAINER
# ─────────────────────────────────────────────────────────────
def train_asr(model: ASRModel,
              train_loader: DataLoader,
              val_loader:   DataLoader,
              tokenizer:    CharTokenizer):

    model.to(DEVICE)
    loss_fn   = nn.CTCLoss(blank=tokenizer.BLANK_IDX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=TC.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    log_path = os.path.join(TC.LOG_DIR, "asr_train.csv")
    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow(["epoch", "train_loss", "val_loss", "val_wer", "lr", "time_s"])

    best_val_loss  = float("inf")
    patience_count = 0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'═'*60}")
    print(f"  ASR Training   |  {params:,} parameters  |  {DEVICE}")
    print(f"{'═'*60}")

    for epoch in range(1, TC.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (feats, labels, in_lens, lbl_lens) in enumerate(train_loader):
            feats   = feats.to(DEVICE)
            labels  = labels.to(DEVICE)
            in_lens = in_lens.to(DEVICE)
            lbl_lens= lbl_lens.to(DEVICE)

            log_probs = model(feats)                       # (B, T, vocab)
            log_probs = log_probs.permute(1, 0, 2)        # (T, B, vocab) for CTC

            loss = loss_fn(log_probs, labels, in_lens, lbl_lens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC.GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                avg = running_loss / (step + 1)
                print(f"  [{epoch:02d}] step {step+1:04d} | loss {avg:.4f}")

        # ── Validation ────────────────────────────────────
        val_loss, val_wer = evaluate_asr(model, val_loader, loss_fn, tokenizer)
        train_loss = running_loss / len(train_loader)
        elapsed    = time.time() - t0
        cur_lr     = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(
            f"\n  Epoch {epoch:02d}/{TC.EPOCHS} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"WER {val_wer*100:.1f}% | LR {cur_lr:.1e} | {elapsed:.0f}s"
        )
        logger.writerow([epoch, train_loss, val_loss, val_wer, cur_lr, elapsed])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_loss, "asr")
        else:
            patience_count += 1
            if patience_count >= TC.PATIENCE:
                print("  🛑  Early stopping.")
                break

    log_file.close()
    print(f"\n  ASR training complete. Logs → {log_path}")


def evaluate_asr(model, loader, loss_fn, tokenizer):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for feats, labels, in_lens, lbl_lens in loader:
            feats   = feats.to(DEVICE)
            labels  = labels.to(DEVICE)
            in_lens = in_lens.to(DEVICE)
            lbl_lens= lbl_lens.to(DEVICE)

            log_probs = model(feats).permute(1, 0, 2)
            loss      = loss_fn(log_probs, labels, in_lens, lbl_lens)
            total_loss += loss.item()

            preds = log_probs.permute(1, 0, 2).argmax(dim=-1)  # (B, T)
            start = 0
            for i, llen in enumerate(lbl_lens):
                ref_ids  = labels[start: start + llen].tolist()
                all_refs.append(tokenizer.decode(ref_ids))
                all_preds.append(tokenizer.ctc_decode(preds[i].tolist()))
                start += llen.item()

    avg_loss = total_loss / len(loader)
    word_err = compute_wer(all_refs, all_preds)
    return avg_loss, word_err


# ─────────────────────────────────────────────────────────────
#  2.  TRANSLATION TRAINER
# ─────────────────────────────────────────────────────────────
def train_translation(model:       TranslatorModel,
                      train_loader: DataLoader,
                      val_loader:   DataLoader,
                      tgt_tokenizer: CharTokenizer):

    model.to(DEVICE)
    pad_idx   = CharTokenizer.PAD_IDX
    loss_fn   = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0,
                           betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, MC_D := 256, TC.WARMUP_STEPS)

    from config import ModelConfig as MC
    log_path = os.path.join(TC.LOG_DIR, "translation_train.csv")
    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow(["epoch", "train_loss", "val_loss", "val_bleu", "time_s"])

    best_val_loss  = float("inf")
    patience_count = 0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'═'*60}")
    print(f"  Translation Training | {params:,} parameters | {DEVICE}")
    print(f"{'═'*60}")

    for epoch in range(1, TC.EPOCHS + 1):
        model.train()
        t0           = time.time()
        running_loss = 0.0

        for step, (src, tgt_in, tgt_out) in enumerate(train_loader):
            src, tgt_in, tgt_out = (
                src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            )
            logits = model(src, tgt_in)          # (B, tgt_len, vocab)
            B, T, V = logits.size()

            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                ppl = math.exp(min(running_loss / (step + 1), 100))
                print(f"  [{epoch:02d}] step {step+1:04d} | "
                      f"loss {running_loss/(step+1):.4f} | ppl {ppl:.1f}")

        val_loss, val_bleu = evaluate_translation(model, val_loader, loss_fn, tgt_tokenizer)
        train_loss = running_loss / len(train_loader)
        elapsed    = time.time() - t0

        print(
            f"\n  Epoch {epoch:02d}/{TC.EPOCHS} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"BLEU {val_bleu:.2f} | {elapsed:.0f}s"
        )
        logger.writerow([epoch, train_loss, val_loss, val_bleu, elapsed])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_loss, "translator")
        else:
            patience_count += 1
            if patience_count >= TC.PATIENCE:
                print("  🛑  Early stopping.")
                break

    log_file.close()
    print(f"\n  Translation training complete. Logs → {log_path}")


def evaluate_translation(model, loader, loss_fn, tgt_tokenizer):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = (
                src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            )
            logits = model(src, tgt_in)
            B, T, V = logits.size()
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))
            total_loss += loss.item()

            preds = logits.argmax(-1)              # (B, tgt_len)
            for i in range(B):
                pred_text = tgt_tokenizer.decode(preds[i].tolist())
                ref_text  = tgt_tokenizer.decode(tgt_out[i].tolist())
                all_preds.append(pred_text)
                all_refs.append([ref_text])

    avg_loss  = total_loss / len(loader)
    bleu      = BLEU()
    bleu_score = bleu.corpus_score(all_preds, all_refs).score
    return avg_loss, bleu_score


# ─────────────────────────────────────────────────────────────
#  3.  TTS TRAINER
# ─────────────────────────────────────────────────────────────
def train_tts(model:        TTSModel,
              train_loader: DataLoader,
              val_loader:   DataLoader):

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=TC.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mel_loss  = nn.MSELoss()
    bce_loss  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))

    log_path = os.path.join(TC.LOG_DIR, "tts_train.csv")
    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow(["epoch", "train_loss", "val_loss", "time_s"])

    best_val_loss  = float("inf")
    patience_count = 0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'═'*60}")
    print(f"  TTS Training  |  {params:,} parameters  |  {DEVICE}")
    print(f"{'═'*60}")

    for epoch in range(1, TC.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (texts, mels, text_lens, mel_lens) in enumerate(train_loader):
            texts = texts.to(DEVICE)
            mels  = mels.to(DEVICE)

            # Build stop targets: 0 for all frames except last → 1
            stop_targets = torch.zeros(mels.size(0), mels.size(2), device=DEVICE)
            for i, l in enumerate(mel_lens):
                stop_targets[i, l - 1] = 1.0

            mel_out, stop_out = model(texts, mels, teacher_forcing=True)

            loss_mel  = mel_loss(mel_out, mels)
            loss_stop = bce_loss(stop_out, stop_targets)
            loss      = loss_mel + loss_stop

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TC.GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  [{epoch:02d}] step {step+1:04d} | "
                      f"loss {running_loss/(step+1):.4f} "
                      f"(mel {loss_mel.item():.4f} + stop {loss_stop.item():.4f})")

        val_loss = evaluate_tts(model, val_loader, mel_loss, bce_loss, DEVICE)
        train_loss = running_loss / len(train_loader)
        elapsed    = time.time() - t0
        scheduler.step()

        print(f"\n  Epoch {epoch:02d}/{TC.EPOCHS} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | {elapsed:.0f}s")
        logger.writerow([epoch, train_loss, val_loss, elapsed])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_loss, "tts")
        else:
            patience_count += 1
            if patience_count >= TC.PATIENCE:
                print("  🛑  Early stopping.")
                break

    log_file.close()
    print(f"\n  TTS training complete. Logs → {log_path}")


def evaluate_tts(model, loader, mel_loss, bce_loss, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for texts, mels, text_lens, mel_lens in loader:
            texts = texts.to(device)
            mels  = mels.to(device)
            stop_targets = torch.zeros(mels.size(0), mels.size(2), device=device)
            for i, l in enumerate(mel_lens):
                stop_targets[i, l - 1] = 1.0
            mel_out, stop_out = model(texts, mels, teacher_forcing=False)
            loss  = mel_loss(mel_out, mels) + bce_loss(stop_out, stop_targets)
            total_loss += loss.item()
    return total_loss / len(loader)

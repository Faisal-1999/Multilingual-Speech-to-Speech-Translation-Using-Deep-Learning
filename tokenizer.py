"""
================================================================
  tokenizer.py  —  Character-level tokenizer supporting
                   English + Hindi (Devanagari) scripts
================================================================
"""

class CharTokenizer:
    """
    Builds vocabulary from training transcripts.
    Supports any Unicode script — works for Hindi, Tamil, etc.

    Special tokens:
        <pad>   → 0   padding
        <sos>   → 1   start-of-sequence  (used by decoder)
        <eos>   → 2   end-of-sequence
        <unk>   → 3   unknown character
        <blank> → 4   CTC blank token   (used by ASR CTC loss)
    """

    PAD_TOKEN   = "<pad>"
    SOS_TOKEN   = "<sos>"
    EOS_TOKEN   = "<eos>"
    UNK_TOKEN   = "<unk>"
    BLANK_TOKEN = "<blank>"

    PAD_IDX   = 0
    SOS_IDX   = 1
    EOS_IDX   = 2
    UNK_IDX   = 3
    BLANK_IDX = 4

    def __init__(self):
        self.char2idx = {
            self.PAD_TOKEN  : self.PAD_IDX,
            self.SOS_TOKEN  : self.SOS_IDX,
            self.EOS_TOKEN  : self.EOS_IDX,
            self.UNK_TOKEN  : self.UNK_IDX,
            self.BLANK_TOKEN: self.BLANK_IDX,
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def build_vocab(self, texts: list[str]):
        """
        Call this once with all transcripts before training.
        texts: list of strings (raw transcripts from dataset)
        """
        unique_chars = sorted(set("".join(texts)))
        for ch in unique_chars:
            if ch not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
        self.vocab_size = len(self.char2idx)
        print(f"[Tokenizer] Vocab built: {self.vocab_size} tokens")
        return self

    def encode(self, text: str, add_sos=False, add_eos=False) -> list[int]:
        """
        'नमस्ते' → [5, 6, 7, 8, 9, 10]
        """
        ids = [self.char2idx.get(c, self.UNK_IDX) for c in text]
        if add_sos:
            ids = [self.SOS_IDX] + ids
        if add_eos:
            ids = ids + [self.EOS_IDX]
        return ids

    def decode(self, indices: list[int], remove_special=True) -> str:
        """
        [5, 6, 7, ...] → 'नमस्ते'
        """
        special = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX,
                   self.UNK_IDX, self.BLANK_IDX}
        chars = []
        for idx in indices:
            if remove_special and idx in special:
                continue
            chars.append(self.idx2char.get(idx, self.UNK_TOKEN))
        return "".join(chars)

    def ctc_decode(self, indices: list[int]) -> str:
        """
        Greedy CTC decode — collapse repeated tokens, remove blanks.
        Used after ASR model output.
        """
        out, prev = [], None
        for idx in indices:
            if idx != prev and idx != self.BLANK_IDX and idx != self.PAD_IDX:
                out.append(self.idx2char.get(idx, ""))
            prev = idx
        return "".join(out)

    def save(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char   = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"[Tokenizer] Loaded {self.vocab_size} tokens from {path}")
        return self

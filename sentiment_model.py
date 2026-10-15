"""بارگذاری مدل ParsBERT (معماری سفارشی Keras) و پیش‌بینی احساس. بدون آموزش و بدون اینترنت."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# اجبار حالت آفلاین؛ باید قبل از import کردن transformers/tensorflow باشد
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from preprocessing import normalize_persian_text

MAX_LENGTH = 128      # طبق InputLayer در config.json
HIDDEN_SIZE = 768     # طبق build_config لایه classifier
NUM_CLASSES = 3       # units=3
VALID_POOLING = ("pooler", "cls", "mean")

DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "PARSBERT_MODEL_DIR",
        Path(__file__).resolve().parent / "ParsBERT-sentiment-fa-finetuned",
    )
)
# تأییدنشده در فایل‌ها: نوع خروجی ۷۶۸‌بعدی BertLayer. پیش‌فرض: pooler
DEFAULT_POOLING = os.environ.get("PARSBERT_POOLING", "pooler").lower()

LABELS_FA = {"Negative": "منفی", "Positive": "مثبت", "Neutral": "خنثی"}


class SentimentError(Exception):
    """پایه خطاهای برنامه (پیام‌ها فارسی و قابل نمایش به کاربرند)."""


class ModelLoadError(SentimentError):
    pass


class InvalidInputError(SentimentError):
    pass


@dataclass
class SentimentResult:
    label: str                 # برچسب نهایی فارسی
    label_en: str              # برچسب اصلی مدل
    scores: dict[str, float]   # امتیاز هر کلاس (جمع = ۱)، با کلید فارسی
    truncated: bool            # آیا متن بیش از ۱۲۸ توکن بود و بریده شد؟


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _load_classifier_weights(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """استخراج kernel (768x3) و bias (3) لایه classifier از فایل h5."""
    import h5py

    kernels, biases = [], []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            lowered = name.lower()
            if "optimizer" in lowered:
                return
            if obj.shape == (HIDDEN_SIZE, NUM_CLASSES):
                kernels.append((name, obj[()]))
            elif obj.shape == (NUM_CLASSES,):
                biases.append((name, obj[()]))

        f.visititems(visit)

    def pick(items, kind):
        if len(items) == 1:
            return items[0][1]
        named = [i for i in items if "classifier" in i[0].lower()]
        if len(named) == 1:
            return named[0][1]
        names = [i[0] for i in items]
        raise ModelLoadError(
            f"وزن {kind} لایه classifier به‌طور یکتا پیدا نشد. موارد یافت‌شده: {names}"
        )

    return pick(kernels, "kernel"), pick(biases, "bias")


class SentimentAnalyzer:
    def __init__(self, model_dir: str | Path | None = None, pooling: str | None = None):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.pooling = (pooling or DEFAULT_POOLING).lower()
        if self.pooling not in VALID_POOLING:
            raise ModelLoadError(f"مقدار pooling باید یکی از {VALID_POOLING} باشد.")
        self._load()

    # ---------- بارگذاری (فقط یک بار، هنگام ساخت شیء) ----------
    def _load(self) -> None:
        base_dir = self.model_dir / "parsbert_base_tf"
        required = [
            self.model_dir / "labels.json",
            self.model_dir / "vocab.txt",
            self.model_dir / "tokenizer_config.json",
            self.model_dir / "parsbert_sentiment.weights.h5",
            base_dir / "config.json",
            base_dir / "tf_model.h5",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise ModelLoadError(
                "فایل‌های لازم مدل پیدا نشد. مسیر پوشه مدل را بررسی کنید.\n"
                + "\n".join(missing)
            )

        # برچسب‌ها از labels.json (ترتیب حدس زده نمی‌شود)
        try:
            raw = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))
            self.id2label = {int(k): v for k, v in raw["id2label"].items()}
        except Exception as exc:
            raise ModelLoadError(f"خواندن labels.json ناموفق بود: {exc}") from exc
        if sorted(self.id2label) != list(range(NUM_CLASSES)) or any(
            v not in LABELS_FA for v in self.id2label.values()
        ):
            raise ModelLoadError("محتوای labels.json با سه برچسب مورد انتظار سازگار نیست.")

        # وزن‌های لایه classifier
        try:
            self.kernel, self.bias = _load_classifier_weights(
                self.model_dir / "parsbert_sentiment.weights.h5"
            )
        except SentimentError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"خواندن وزن‌های طبقه‌بند ناموفق بود: {exc}") from exc

        # توکنایزر و بدنه BERT (کاملاً محلی)
        try:
            import tensorflow as tf
            from transformers import BertTokenizerFast, TFBertModel
        except Exception as exc:
            raise ModelLoadError(
                "کتابخانه‌های tensorflow/transformers نصب یا سازگار نیستند. "
                f"requirements.txt را نصب کنید. جزئیات: {exc}"
            ) from exc

        try:
            self._tf = tf
            self.tokenizer = BertTokenizerFast.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self.bert = TFBertModel.from_pretrained(str(base_dir), local_files_only=True)
        except Exception as exc:
            raise ModelLoadError(f"بارگذاری BERT یا توکنایزر ناموفق بود: {exc}") from exc

        # اجرای آزمایشی کوچک برای ساخته‌شدن گراف
        self._forward("سلام")

    # ---------- استنتاج ----------
    def _forward(self, text: str, pooling: str | None = None):
        pooling = pooling or self.pooling
        tf = self._tf
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        input_ids = tf.convert_to_tensor(enc["input_ids"], dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(enc["attention_mask"], dtype=tf.int32)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=False)

        if pooling == "pooler":
            pooled = out.pooler_output
        elif pooling == "cls":
            pooled = out.last_hidden_state[:, 0, :]
        else:  # mean
            mask = tf.cast(attention_mask, tf.float32)[:, :, None]
            pooled = tf.reduce_sum(out.last_hidden_state * mask, axis=1) / tf.reduce_sum(
                mask, axis=1
            )
        pooled = pooled.numpy().astype(np.float32)
        logits = pooled @ self.kernel + self.bias   # معادل Dense
        return _softmax(logits)[0]

    def analyze(self, text: str, pooling: str | None = None) -> SentimentResult:
        """متن فارسی را می‌گیرد و برچسب نهایی و امتیاز هر سه کلاس را برمی‌گرداند."""
        if not isinstance(text, str):
            raise InvalidInputError("ورودی باید متن باشد.")
        clean = normalize_persian_text(text)
        if not clean:
            raise InvalidInputError("متن خالی است. لطفاً یک متن فارسی وارد کنید.")
        if pooling is not None and pooling not in VALID_POOLING:
            raise InvalidInputError(f"pooling نامعتبر است: {pooling}")

        n_tokens = len(self.tokenizer(clean, truncation=False)["input_ids"])
        try:
            probs = self._forward(clean, pooling)
        except Exception as exc:
            raise SentimentError(f"خطا در اجرای مدل: {exc}") from exc

        scores = {LABELS_FA[self.id2label[i]]: float(probs[i]) for i in range(NUM_CLASSES)}
        best = int(np.argmax(probs))
        return SentimentResult(
            label=LABELS_FA[self.id2label[best]],
            label_en=self.id2label[best],
            scores=scores,
            truncated=n_tokens > MAX_LENGTH,
        )
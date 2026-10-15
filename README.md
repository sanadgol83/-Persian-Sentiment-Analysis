<div align="center">

# 💬 Persian Sentiment Analysis
# سامانه تحلیل احساسات متون فارسی

**ParsBERT · TensorFlow · Streamlit · Offline · CPU-friendly**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-FF6F00?logo=tensorflow&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.40.2-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?logo=streamlit&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20CPU-0078D6)

[🇬🇧 English](#english) &nbsp;|&nbsp; [🇮🇷 فارسی](#persian)

</div>

---

<a id="english"></a>

# 🇬🇧 English

## Overview

A small web app that classifies Persian text as **Positive**, **Negative** or **Neutral**. It uses an already fine-tuned model, [`Rasooli/ParsBERT-sentiment-fa-finetuned`](https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned), and a Streamlit interface with right-to-left support.

- ❌ **No training, no fine-tuning, no dataset download.** The app only loads the existing weights and runs inference.
- 🔌 **Works offline** once the model files are on disk.
- 🖥️ **Runs on CPU** (tested on Windows with 8 GB RAM, no dedicated GPU).
- ⚡ The model is loaded **once** and cached between requests.

## Features

- Persian UI with a large text box and an "analyze" button
- Final label plus a score for each of the three classes
- Input validation and readable error messages
- Minimal text normalization (Arabic ي/ك → Persian ی/ک, extra spaces). Punctuation, emoji, half-spaces and negation words are **kept**

## How it works

The model is a custom Keras architecture, so it cannot be loaded with `transformers.pipeline`. Based on the files in the model repository:

```
input_ids (128) ─┐
                 ├─► BERT (ParsBERT, frozen) ─► 768-d vector ─► Dense(3, softmax)
attention_mask ──┘
```

- The BERT body is loaded from `parsbert_base_tf/` with `TFBertModel`.
- The weights of the final `Dense` layer are read from `parsbert_sentiment.weights.h5`.
- Label order comes from `labels.json`: `0 = Negative`, `1 = Positive`, `2 = Neutral`.
- Maximum input length is **128 tokens**; longer texts are truncated (the UI warns about this).

> **Technical note:** the original `BertLayer` source code is not included in the model repository, so it is not documented how the 768-d vector is obtained from BERT. This project uses the `pooler` output, chosen empirically after comparing `pooler`, `cls` and `mean` pooling on sample sentences (`check_pooling.py`). It is a well-supported choice, not an officially confirmed one. You can override it with the `PARSBERT_POOLING` environment variable.

## Project structure

```
persian-sentiment-analysis/
├── app.py                        # Streamlit user interface
├── sentiment_model.py            # Model loading + prediction
├── preprocessing.py              # Minimal Persian text normalization
├── check_pooling.py              # Optional diagnostic script
├── requirements.txt
├── README.md
├── .gitignore
└── ParsBERT-sentiment-fa-finetuned/   # ⚠️ NOT in this repo – download it (see below)
```

## Getting started

### 1. Prerequisites

- Windows (other OSes are untested), **Python 3.10** (TensorFlow 2.15 does not support Python 3.12+)
- [Git](https://git-scm.com/) (with Git LFS / Xet support, which is included in recent versions)
- About 3 GB of free disk space and 8 GB RAM

```bat
py -0
py -3.10 --version
```

### 2. Clone this repository

```bat
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 3. Download the model

The model files are **not** stored in this repository (about 650 MB). See [Model download](#model-download) below, then come back here.

### 4. Create and activate a virtual environment

**CMD**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python --version
```

**PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 5. Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation:

```bat
python -c "import tensorflow as tf, transformers, h5py, streamlit; print(tf.__version__, transformers.__version__)"
```

Expected output ends with `2.15.1 4.40.2`.

### 6. Run

```bat
streamlit run app.py
```

Open <http://localhost:8501>. The first analysis takes a few extra seconds while the model loads.

## Model download

The model lives on Hugging Face: **<https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned>**

It must end up in a folder named **`ParsBERT-sentiment-fa-finetuned`** next to `app.py`.

**Option A – git clone (recommended)**

Run this inside the project folder:

```bat
git clone https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned
```

**Option B – manual download**

1. Open the model page → **Files and versions**.
2. Download every file, including the `parsbert_base_tf/` subfolder.
3. Put them in `ParsBERT-sentiment-fa-finetuned/` next to `app.py`.

**Option C – model stored elsewhere**

Keep the folder anywhere and point the app to it:

```bat
:: CMD
set PARSBERT_MODEL_DIR=D:\models\ParsBERT-sentiment-fa-finetuned
```

```powershell
# PowerShell
$env:PARSBERT_MODEL_DIR = "D:\models\ParsBERT-sentiment-fa-finetuned"
```

**Expected folder content**

```
ParsBERT-sentiment-fa-finetuned/
├── labels.json
├── parsbert_base_tf/
│   ├── config.json
│   └── tf_model.h5                  (~650 MB)
├── parsbert_sentiment.keras
├── parsbert_sentiment.weights.h5
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── README.md
```

> Check that `tf_model.h5` is about 650 MB. If it is only a few KB, the large files were not downloaded; run `git lfs pull` inside the model folder.

The model is subject to the license shown on its Hugging Face page. Please check it before redistributing or using it commercially.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `PARSBERT_MODEL_DIR` | `./ParsBERT-sentiment-fa-finetuned` | Path to the model folder |
| `PARSBERT_POOLING` | `pooler` | `pooler`, `cls` or `mean` (see the technical note) |

The app forces `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`, so nothing is downloaded at runtime.

## Manual test samples

| Text | Expected |
|---|---|
| این محصول کیفیت بسیار خوبی دارد و از خریدم راضی هستم. | Positive |
| کیفیت محصول بسیار بد بود و اصلاً ارزش خرید نداشت. | Negative |
| امروز بسته را تحویل گرفتم. | Probably Neutral |
| خوب نبود. | Probably Negative |
| اصلاً انتظار چنین کیفیت خوبی را نداشتم. | Ambiguous (see limitations) |

These are for manual testing only; they are not training data and results are not guaranteed.

## Limitations

- Sarcasm, ambiguous sentences and "surprise" constructions can be misclassified. For example, the sentence *«اصلاً انتظار چنین کیفیت خوبی را نداشتم»* ("I never expected such good quality") is typically read as positive by humans, but the model classified it as strongly negative in testing.
- The percentages are **softmax outputs**. They sum to 100 % but are **not** the real probability that the prediction is correct; the model can be confidently wrong.
- Texts longer than 128 tokens are truncated.
- The training data of the model is not documented in its repository, so its performance on your domain should be verified with your own labeled examples.

## Troubleshooting

| Problem | Solution |
|---|---|
| `فایل‌های لازم مدل پیدا نشد` / missing files | Check the model folder name and location, or set `PARSBERT_MODEL_DIR` |
| `tf_model.h5` is tiny | Run `git lfs pull` inside the model folder |
| No matching distribution for TensorFlow | Use Python 3.10 (not 3.12+) |
| `.ps1` file opens in an editor (CMD) | In CMD use `.venv\Scripts\activate.bat` |
| `pip` fails with `NameResolutionError` for `files.pythonhosted.org` | Your network blocks that domain. Use another network/VPN, or a PyPI mirror: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` (a third-party mirror; use it only if needed) |
| `DLL load failed` from TensorFlow | Install the *Microsoft Visual C++ Redistributable 2015–2022* |
| Path-too-long errors on Windows | Enable Long Paths or move the project to a shorter path |
| TensorFlow prints oneDNN / deprecation warnings | Harmless |

## Acknowledgements

- Model: [Rasooli/ParsBERT-sentiment-fa-finetuned](https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned)
- Base language model: ParsBERT by [HooshvareLab](https://huggingface.co/HooshvareLab)
- [Streamlit](https://streamlit.io/), [Hugging Face Transformers](https://github.com/huggingface/transformers), [TensorFlow](https://www.tensorflow.org/)

## License

The code in this repository is released under the license in the `LICENSE` file (add one before publishing). The model files are **not** part of this repository and follow the license of their original Hugging Face page.

---

<a id="persian"></a>

<div dir="rtl" align="right">

# 🇮🇷 فارسی

## معرفی

یک برنامه وب کوچک که متن فارسی را در یکی از سه دسته‌ی **مثبت**، **منفی** یا **خنثی** قرار می‌دهد. برنامه از مدل ازپیش‌آموزش‌دیده‌ی [`Rasooli/ParsBERT-sentiment-fa-finetuned`](https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned) و یک رابط کاربری Streamlit با پشتیبانی از راست‌چین استفاده می‌کند.

- ❌ **بدون آموزش، بدون Fine-tuning و بدون دانلود دیتاست.** برنامه فقط وزن‌های موجود را بارگذاری می‌کند و پیش‌بینی انجام می‌دهد.
- 🔌 **بدون نیاز به اینترنت** پس از آنکه فایل‌های مدل روی دیسک قرار گرفتند.
- 🖥️ **اجرا روی CPU** (آزمون‌شده روی ویندوز با ۸ گیگابایت RAM و بدون کارت گرافیک مجزا).
- ⚡ مدل فقط **یک بار** بارگذاری و در درخواست‌های بعدی استفاده می‌شود.

## قابلیت‌ها

- رابط فارسی با کادر متنی بزرگ و دکمه «تحلیل احساسات»
- نمایش برچسب نهایی و امتیاز هر سه کلاس
- بررسی ورودی و نمایش پیام خطای قابل‌فهم
- پیش‌پردازش حداقلی (تبدیل «ي» و «ك» عربی به «ی» و «ک» فارسی و حذف فاصله‌های اضافی). علائم نگارشی، ایموجی، نیم‌فاصله و کلمات منفی **حذف نمی‌شوند**

## نحوه کار

مدل یک معماری سفارشی Keras است و با `transformers.pipeline` بارگذاری نمی‌شود. بر اساس فایل‌های مخزن مدل:

```
input_ids (128) ─┐
                 ├─► BERT (ParsBERT، فریز) ─► بردار ۷۶۸‌بعدی ─► Dense(3, softmax)
attention_mask ──┘
```

- بدنه BERT با `TFBertModel` از پوشه `parsbert_base_tf/` بارگذاری می‌شود.
- وزن‌های لایه نهایی `Dense` از فایل `parsbert_sentiment.weights.h5` خوانده می‌شود.
- ترتیب برچسب‌ها از `labels.json` گرفته می‌شود: `0 = منفی`، `1 = مثبت`، `2 = خنثی`.
- حداکثر طول ورودی **۱۲۸ توکن** است و متن‌های بلندتر بریده می‌شوند (رابط کاربری هشدار می‌دهد).

> **نکته فنی:** کد اصلی `BertLayer` در مخزن مدل موجود نیست، پس مستند نشده که بردار ۷۶۸‌بعدی چگونه از BERT گرفته شده است. این پروژه از خروجی `pooler` استفاده می‌کند که پس از مقایسه‌ی سه حالت `pooler`، `cls` و `mean` روی جمله‌های نمونه (با `check_pooling.py`) انتخاب شده است. این انتخاب مستدل است، اما رسماً تأیید نشده. با متغیر محیطی `PARSBERT_POOLING` می‌توانید آن را تغییر دهید.

## ساختار پروژه

```
persian-sentiment-analysis/
├── app.py                        # رابط کاربری Streamlit
├── sentiment_model.py            # بارگذاری مدل و پیش‌بینی
├── preprocessing.py              # پیش‌پردازش حداقلی متن فارسی
├── check_pooling.py              # اسکریپت تشخیصی (اختیاری)
├── requirements.txt
├── README.md
├── .gitignore
└── ParsBERT-sentiment-fa-finetuned/   # ⚠️ در این مخزن نیست – باید دانلود شود (پایین‌تر)
```

## شروع کار

### ۱. پیش‌نیازها

- ویندوز (سایر سیستم‌عامل‌ها آزمایش نشده‌اند)، **Python 3.10** (TensorFlow 2.15 از Python 3.12 به بالا پشتیبانی نمی‌کند)
- [Git](https://git-scm.com/) (نسخه‌های جدید از Git LFS / Xet پشتیبانی می‌کنند)
- حدود ۳ گیگابایت فضای خالی دیسک و ۸ گیگابایت RAM

```bat
py -0
py -3.10 --version
```

### ۲. کلون کردن این مخزن

```bat
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### ۳. دانلود مدل

فایل‌های مدل (حدود ۶۵۰ مگابایت) در این مخزن **نیستند**. بخش [دانلود مدل](#persian-model) را در پایین ببینید و سپس به همین‌جا برگردید.

### ۴. ساخت و فعال‌سازی محیط مجازی

**CMD**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python --version
```

**PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

اگر PowerShell اجرای اسکریپت را مسدود کرد: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### ۵. نصب کتابخانه‌ها

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

بررسی نصب:

```bat
python -c "import tensorflow as tf, transformers, h5py, streamlit; print(tf.__version__, transformers.__version__)"
```

خروجی باید به `2.15.1 4.40.2` ختم شود.

### ۶. اجرا

```bat
streamlit run app.py
```

آدرس <http://localhost:8501> را باز کنید. در اولین تحلیل، چند ثانیه صبر لازم است تا مدل بارگذاری شود.

<a id="persian-model"></a>

## دانلود مدل

مدل در Hugging Face قرار دارد: **<https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned>**

مدل باید در پوشه‌ای با نام **`ParsBERT-sentiment-fa-finetuned`** کنار `app.py` قرار بگیرد.

**روش الف – git clone (پیشنهادی)**

این دستور را داخل پوشه پروژه اجرا کنید:

```bat
git clone https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned
```

**روش ب – دانلود دستی**

۱. صفحه مدل ← تب **Files and versions** را باز کنید.
۲. همه فایل‌ها را، از جمله زیرپوشه `parsbert_base_tf/`، دانلود کنید.
۳. آن‌ها را در پوشه `ParsBERT-sentiment-fa-finetuned/` کنار `app.py` بگذارید.

**روش ج – مدل در مسیر دیگری باشد**

پوشه را هرجا که خواستید نگه دارید و مسیر آن را به برنامه بدهید:

```bat
:: CMD
set PARSBERT_MODEL_DIR=D:\models\ParsBERT-sentiment-fa-finetuned
```

```powershell
# PowerShell
$env:PARSBERT_MODEL_DIR = "D:\models\ParsBERT-sentiment-fa-finetuned"
```

**ساختار مورد انتظار پوشه مدل**

```
ParsBERT-sentiment-fa-finetuned/
├── labels.json
├── parsbert_base_tf/
│   ├── config.json
│   └── tf_model.h5                  (حدود ۶۵۰ مگابایت)
├── parsbert_sentiment.keras
├── parsbert_sentiment.weights.h5
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── README.md
```

> بررسی کنید که حجم `tf_model.h5` حدود ۶۵۰ مگابایت باشد. اگر فقط چند کیلوبایت است، فایل‌های بزرگ دانلود نشده‌اند؛ داخل پوشه مدل دستور `git lfs pull` را اجرا کنید.

استفاده از مدل تابع مجوزی است که در صفحه Hugging Face آن آمده است. پیش از بازنشر یا استفاده تجاری آن را بررسی کنید.

## تنظیمات

| متغیر محیطی | مقدار پیش‌فرض | کاربرد |
|---|---|---|
| `PARSBERT_MODEL_DIR` | `./ParsBERT-sentiment-fa-finetuned` | مسیر پوشه مدل |
| `PARSBERT_POOLING` | `pooler` | یکی از `pooler`، `cls` یا `mean` (نکته فنی بالا را ببینید) |

برنامه مقادیر `TRANSFORMERS_OFFLINE=1` و `HF_HUB_OFFLINE=1` را اجبار می‌کند، پس هنگام اجرا چیزی دانلود نمی‌شود.

## نمونه‌های آزمایش دستی

| متن | نتیجه مورد انتظار |
|---|---|
| این محصول کیفیت بسیار خوبی دارد و از خریدم راضی هستم. | مثبت |
| کیفیت محصول بسیار بد بود و اصلاً ارزش خرید نداشت. | منفی |
| امروز بسته را تحویل گرفتم. | احتمالاً خنثی |
| خوب نبود. | احتمالاً منفی |
| اصلاً انتظار چنین کیفیت خوبی را نداشتم. | مبهم (بخش محدودیت‌ها) |

این نمونه‌ها فقط برای آزمایش دستی‌اند، داده آموزشی نیستند و نتیجه‌ی آن‌ها تضمین‌شده نیست.

## محدودیت‌ها

- جمله‌های طعنه‌دار، مبهم و ساختارهای «شگفتی» ممکن است اشتباه تشخیص داده شوند. مثلاً جمله‌ی «اصلاً انتظار چنین کیفیت خوبی را نداشتم» معمولاً برای انسان مثبت است، اما مدل در آزمایش آن را با اطمینان بالا منفی تشخیص داد.
- درصدها خروجی **softmax** هستند. جمع آن‌ها ۱۰۰٪ است، ولی **احتمال واقعی** درست‌بودن پیش‌بینی نیستند و مدل می‌تواند با اطمینان ظاهری بالا اشتباه کند.
- متن‌های بیش از ۱۲۸ توکن بریده می‌شوند.
- داده آموزشی مدل در مخزن آن مستند نشده است؛ بنابراین کیفیت آن را روی حوزه‌ی خودتان با نمونه‌های برچسب‌خورده‌ی خودتان بسنجید.

## رفع خطاهای رایج

| مشکل | راه‌حل |
|---|---|
| پیام «فایل‌های لازم مدل پیدا نشد» | نام و محل پوشه مدل را بررسی کنید یا `PARSBERT_MODEL_DIR` را تنظیم کنید |
| حجم `tf_model.h5` خیلی کم است | داخل پوشه مدل `git lfs pull` را اجرا کنید |
| خطای نصب TensorFlow (No matching distribution) | از Python 3.10 استفاده کنید (نه 3.12 به بالا) |
| فایل `.ps1` در CMD با ویرایشگر باز می‌شود | در CMD از `.venv\Scripts\activate.bat` استفاده کنید |
| خطای `NameResolutionError` برای `files.pythonhosted.org` در pip | شبکه شما این دامنه را مسدود کرده است. از شبکه یا VPN دیگر استفاده کنید، یا از آینه PyPI: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` (آینه شخص ثالث است؛ فقط در صورت نیاز) |
| خطای `DLL load failed` در TensorFlow | بسته «Microsoft Visual C++ Redistributable 2015–2022» را نصب کنید |
| خطای طولانی بودن مسیر در ویندوز | Long Paths را فعال کنید یا پروژه را در مسیر کوتاه‌تری بگذارید |
| هشدارهای oneDNN و deprecation در TensorFlow | بی‌خطرند |

## قدردانی

- مدل: [Rasooli/ParsBERT-sentiment-fa-finetuned](https://huggingface.co/Rasooli/ParsBERT-sentiment-fa-finetuned)
- مدل زبانی پایه: ParsBERT از [HooshvareLab](https://huggingface.co/HooshvareLab)
- [Streamlit](https://streamlit.io/)، [Hugging Face Transformers](https://github.com/huggingface/transformers) و [TensorFlow](https://www.tensorflow.org/)

## مجوز

کد این مخزن تحت مجوز فایل `LICENSE` منتشر می‌شود (پیش از انتشار یکی اضافه کنید). فایل‌های مدل بخشی از این مخزن **نیستند** و تابع مجوز صفحه اصلی خود در Hugging Face هستند.

</div>

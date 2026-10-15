from sentiment_model import SentimentAnalyzer, VALID_POOLING

SAMPLES = [
    "این محصول کیفیت بسیار خوبی دارد و از خریدم راضی هستم.",
    "کیفیت محصول بسیار بد بود و اصلاً ارزش خرید نداشت.",
    "امروز بسته را تحویل گرفتم.",
    "خوب نبود.",
    "فیلم فوق‌العاده‌ای بود، عاشقش شدم!",
    "خدمات افتضاح بود، پولم را هدر دادم.",
]

analyzer = SentimentAnalyzer()
for mode in VALID_POOLING:
    print(f"\n===== pooling = {mode} =====")
    for s in SAMPLES:
        r = analyzer.analyze(s, pooling=mode)
        scores = " | ".join(f"{k}:{v*100:5.1f}%" for k, v in r.scores.items())
        print(f"{r.label:>5}  [{scores}]  {s}")
import streamlit as st

from sentiment_model import InvalidInputError, SentimentAnalyzer, SentimentError

st.set_page_config(page_title="تحلیل احساسات متون فارسی", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stApp, .stMarkdown, label, p, h1, h2, h3, li {
        direction: rtl; text-align: right;
        font-family: Tahoma, "Segoe UI", sans-serif;
    }
    textarea { direction: rtl !important; text-align: right !important;
               font-family: Tahoma, "Segoe UI", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="در حال بارگذاری مدل (فقط بار اول)...")
def get_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


st.title("سامانه تحلیل احساسات متون فارسی")
st.write(
    "متن فارسی خود را وارد کنید تا مدل ParsBERT تشخیص دهد احساس غالب آن "
    "**مثبت**، **منفی** یا **خنثی** است. نظرات کاربران، پیام‌ها و متن‌های کوتاه از موارد کاربرد آن‌هاست."
)

text = st.text_area("متن فارسی:", height=200, placeholder="متن خود را اینجا بنویسید...")
clicked = st.button("تحلیل احساسات", type="primary")

if clicked:
    if not text.strip():
        st.warning("لطفاً ابتدا یک متن وارد کنید.")
        st.stop()

    try:
        analyzer = get_analyzer()
    except SentimentError as exc:
        st.error(f"بارگذاری مدل ناموفق بود:\n\n{exc}")
        st.stop()
    except Exception as exc:  # خطای پیش‌بینی‌نشده
        st.error("خطای پیش‌بینی‌نشده هنگام بارگذاری مدل رخ داد.")
        st.exception(exc)
        st.stop()

    try:
        with st.spinner("در حال تحلیل..."):
            result = analyzer.analyze(text)
    except InvalidInputError as exc:
        st.warning(str(exc))
        st.stop()
    except SentimentError as exc:
        st.error(str(exc))
        st.stop()

    if result.label == "مثبت":
        st.success(f"نتیجه: {result.label} 😊")
    elif result.label == "منفی":
        st.error(f"نتیجه: {result.label} 😞")
    else:
        st.info(f"نتیجه: {result.label} 😐")

    if result.truncated:
        st.caption("⚠️ متن طولانی بود و فقط ۱۲۸ توکن اول آن تحلیل شد.")

    st.subheader("امتیاز هر دسته")
    for name in ("مثبت", "منفی", "خنثی"):
        value = result.scores[name]
        st.write(f"{name}: {value * 100:.1f}٪")
        st.progress(min(max(value, 0.0), 1.0))

    st.caption(
        "این درصدها خروجی softmax مدل‌اند و جمع آن‌ها ۱۰۰٪ است؛ لزوماً احتمال واقعی "
        "درست‌بودن پیش‌بینی نیستند. در جمله‌های مبهم یا طعنه‌دار مدل می‌تواند اشتباه کند."
    )
# YouTube ChatBot

شات بوت بيخليك تحط لينك أي فيديو على يوتيوب وتسأل عن محتواه (RAG بـ LangChain + Groq + FAISS).

## التشغيل على جهازك

```bash
pip install -r requirements.txt
```

اعمل فولدر `.streamlit` وجواه ملف `secrets.toml`:

```toml
GROQ_API_KEY = "gsk_..."
```

وبعدين:

```bash
streamlit run app.py
```

## الرفع على Streamlit Community Cloud (مجاني)

1. ارفع الفولدر ده على GitHub (repo جديد). **متعملش push لملف `secrets.toml`**.
2. ادخل على https://share.streamlit.io وسجّل بحساب GitHub.
3. اضغط **Create app**، واختار الـ repo، والـ Main file يكون `app.py`.
4. من **Advanced settings → Secrets** حط:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. اضغط **Deploy**. هتاخد لينك زي `https://your-app.streamlit.app` تبعته لأي حد.

## مشكلة مهمة: يوتيوب بيحظر السيرفرات

يوتيوب بيحظر أغلب الـ IPs بتاعة السيرفرات (AWS و Google Cloud وغيرهم). فالتطبيق ممكن يشتغل عندك على الجهاز
ويطلع `RequestBlocked` أو `IpBlocked` بعد الرفع.

الحل: proxy من نوع "Residential" من Webshare (https://www.webshare.io):

1. اعمل حساب واشترك في باقة **Residential** (مش Proxy Server ولا Static Residential).
2. من صفحة Proxy Settings خد الـ Username والـ Password.
3. ضيفهم في الـ Secrets:
   ```toml
   WEBSHARE_PROXY_USERNAME = "..."
   WEBSHARE_PROXY_PASSWORD = "..."
   ```

التطبيق هيستخدم الـ proxy لوحده لما يلاقي البيانات دي.

## ملاحظات

- كل اللي هيستخدموا التطبيق هيصرفوا من رصيد مفتاح Groq بتاعك، ولو الاستخدام كتر هيظهرلهم rate limit.
- أول تشغيل بياخد وقت لأن موديل الـ embeddings بيتنزّل (حوالي 500MB).

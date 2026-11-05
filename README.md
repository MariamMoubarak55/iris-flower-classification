# Iris Flower Classification - Production ML Project

مشروع متكامل لتصنيف زهور السوسن (Iris Flower Classification) مبني وفق أفضل الممارسات الهندسية لنقل نماذج التعلم الآلي من بيئة التجارب إلى الإنتاج (From Notebook to Production Deployment).

---

## 🏗️ Project Architecture (هيكل المشروع)

```text
iris_classification_project/
├── .gitignore               # تجاهل ملفات البيئة الافتراضية والـ cache والـ logs
├── README.md                # دليل استخدام وتوثيق المشروع
├── requirements.txt         # تبعات بايثون
├── Dockerfile               # بناء صورة دوكر للتطبيق
├── docker-compose.yml       # تشغيل الحاوية بسهولة
├── run.py                   # نقطة تشغيل التدريب والتجربة محلياً
├── run_app.py               # نقطة تشغيل سيرفر Flask
├── logs.log                 # ملف حفظ سجلات الأحداث (Logging)
│
├── data/                    # مجلد حفظ البيانات (iris.csv)
│
├── src/
│   ├── conf/                # إعدادات المشروع (json-driven)
│   │   ├── design.json      # المعاملات (Hyperparameters, CV folds, paths)
│   │   └── config.py        # دالة get_default() لتحميل الإعدادات ديناميكياً
│   │
│   ├── utils/               # أدوات مساعدة
│   │   └── logger.py        # نظام التتبع والـ Logging الاحترافي
│   │
│   ├── main/                # منطق الـ Machine Learning
│   │   ├── data_loader.py   # تحميل وتجهيز البيانات
│   │   ├── preprocessing.py # التقسيم الطبقي (Stratified Split) ومعالجة الخصائص
│   │   ├── train.py         # ضبط المعاملات عبر GridSearchCV والتدريب وحفظ النموذج
│   │   └── predict.py       # قراءة النموذج واستخراج التوقعات والاحتماليات
│   │
│   ├── resources/           # حفظ النموذج المدرب (model.joblib)
│   │
│   └── template_app/        # تطبيق الـ API باستخدام Flask
│       └── app.py           # الـ Endpoints لاستقبال البيانات وإرجاع التوقعات
│
└── tests/                   # ملفات اختبارات الوحدة (Unit Tests)
    ├── test_data.py         # اختبار جودة وتقسيم البيانات
    ├── test_model.py        # اختبار دقة النموذج وسلامة التنبؤ
    └── test_app.py          # اختبار واجهات الـ REST API
```

---

## 🚀 كيفية التثبيت والتشغيل (Quick Start)

### 1. إعداد البيئة الافتراضية وتثبيت التبعات
```bash
# إنشاء بيئة افتراضية
python -m venv venv

# تفعيل البيئة (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# تثبيت المكتبات
pip install -r requirements.txt
```

### 2. تدريب وتقييم الموديل محلياً
```bash
python run.py
```
يقوم هذا الأمر بـ:
1. جلب/تحميل بيانات Iris.
2. تقسيم البيانات stratified train/test split.
3. تدريب النموذج مع `GridSearchCV` والـ Cross Validation.
4. حساب مقاييس التقييم (Accuracy & Classification Report).
5. حفظ النموذج في `src/resources/model.joblib`.
6. اختبار عينة تنبؤ سريعة.

### 3. تشغيل الاختبارات الآلية (Testing Suite)
```bash
pytest
```

### 4. تشغيل خادم الـ API عبر Flask
```bash
python run_app.py
```
يعمل السيرفر على: `http://localhost:5000`

---

## 📡 واجهات الـ API (Endpoints)

### 1. فحص الحالة (Health Check)
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. إرسال طلب تنبؤ (Predict Endpoint)
```http
POST /predict
Content-Type: application/json
```

**مثال على مدخلات بصيغة قائمة (List of features):**
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
  ]
}
```

**مثال على مدخلات بمفردها (Dictionary):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**نموذج الاستجابة (Output):**
```json
{
  "status": "success",
  "predictions": [
    {
      "sample_index": 0,
      "prediction_id": 0,
      "prediction_name": "setosa",
      "confidence": 0.985,
      "probabilities": {
        "setosa": 0.985,
        "versicolor": 0.014,
        "virginica": 0.001
      }
    }
  ]
}
```

---

## 🐳 التشغيل باستخدام Docker

```bash
# بناء الصورة وتشغيل الحاوية
docker compose up --build
```

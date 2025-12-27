# پروژه طبقه‌بندی اخبار با Multinomial Naive Bayes


الگوریتم استفاده شده: **Multinomial Naive Bayes** با بردارسازی متن با **CountVectorizer**.  

مراحل انجام پروژه:  
1. وارد کردن کتابخانه‌ها  
2. بارگذاری داده‌ها  
3. نمونه‌گیری استراتیفای شده  
4. آماده‌سازی ویژگی‌ها و برچسب‌ها  
5. بردارسازی متن  
6. آموزش مدل  
7. پیش‌بینی  
8. محاسبه معیارهای ارزیابی  
9. نمایش ماتریس سردرگمی  
10. نمودار مقایسه معیارها  
11. پیش‌بینی متن دلخواه

# بارگذاری کتابخانه ها

در این بخش، تمام کتابخانه‌های مورد نیاز برای پروژه وارد می‌شوند.

کتابخانه pandas و numpy برای مدیریت داده‌ها و محاسبات عددی استفاده می‌شوند.

کتابخانه matplotlib و seaborn برای رسم نمودارها، و sklearn برای بردارسازی متن، مدل Naive Bayes و محاسبه معیارهای ارزیابی کاربرد دارند.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
```
## بارگذاری داده‌ها
در این بخش، دیتاست آموزش و تست را از فایل CSV می‌خوانیم.  
همچنین شکل داده‌ها و چند نمونه اول بررسی می‌شود.
```
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample rows:\n", train_df.head())
```
# output:

```
### توضیح خروجی داده‌ها

- **Train shape: (120000, 2)**  
  داده‌های آموزشی شامل ۱۲۰,۰۰۰ نمونه هستند و هر نمونه دو ستون دارد:  
  1. `text` → متن خبر  
  2. `label` → برچسب کلاس متن

- **Test shape: (7600, 2)**  
  داده‌های آزمون شامل ۷,۶۰۰ نمونه هستند و ساختار مشابه داده‌های آموزشی دارند.

```


> مشاهده می‌کنیم که داده‌ها شامل اخبار اقتصادی هستند و برچسب‌ها نشان می‌دهند که هر متن به کدام کلاس تعلق دارد. 

## نمونه‌گیری استراتیفای شده (Stratified Subsample)
برای سریع‌تر شدن آموزش، تعداد نمونه‌ها محدود می‌شود اما نسبت کلاس‌ها حفظ می‌شود.  
- 5000 نمونه از هر کلاس برای آموزش  
- 1000 نمونه از هر کلاس برای تست
```
train_df = train_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(5000, len(x)), random_state=42)
)
test_df = test_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(1000, len(x)), random_state=42)
)

print("Subsampled Train:", train_df.shape)
print("Subsampled Test:", test_df.shape)
```
## آماده‌سازی ویژگی‌ها و برچسب‌ها
ستون متن‌ها به عنوان ورودی (X) و ستون برچسب‌ها به عنوان خروجی (y) تعریف می‌شوند.

```
X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]
```
## بردارسازی متن
متن‌ها به یک بردار عددی تبدیل می‌شوند تا مدل Naive Bayes بتواند از آن‌ها استفاده کند.
```
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```
## پیش‌بینی روی داده‌های تست
- پیش‌بینی کلاس‌ها  
- پیش‌بینی احتمال تعلق هر متن به هر کلاس
```
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)
print("Prediction done.")
```
## محاسبه معیارهای ارزیابی
- Accuracy: درصد پیش‌بینی درست  
- Precision: دقت پیش‌بینی‌های مثبت  
- Recall: حساسیت مدل  
- F1-score: ترکیبی از Precision و Recall

```
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nEvaluation Metrics:")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
# output:

```
Evaluation Metrics:
Accuracy : 0.8995
Precision: 0.8993429511091425
Recall   : 0.8995
F1-score : 0.8993781805271808
```

## ماتریس سردرگمی (Normalized Confusion Matrix)
نشان می‌دهد هر کلاس چه تعداد درست و اشتباه پیش‌بینی شده است.

```
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels_text = ["World", "Sports", "Business", "Sci/Tech"]

plt.figure(figsize=(7,6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels_text, yticklabels=labels_text)
plt.title("Normalized Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
```
# output:

<img width="579" height="547" alt="image" src="https://github.com/user-attachments/assets/b1bfa225-5f1b-48c3-841e-425b0d3054fd" />

## نمودار مقایسه معیارها
نمودار میله‌ای برای مقایسه Accuracy، Precision، Recall و F1-score

```
metrics_values = [acc, prec, rec, f1]
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]

plt.figure(figsize=(8,5))
y_min = min(metrics_values) - 0.02  
y_max = max(metrics_values) + 0.05 
plt.ylim(y_min, y_max)

colors = ['skyblue', 'orange', 'green', 'red']
bars = plt.bar(metrics_names, metrics_values, color=colors)

for bar, value in zip(bars, metrics_values):
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        value + 0.002, 
        f"{value:.3f}", 
        ha='center', 
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Model Performance Metrics (CountVectorizer + Stratified Sampling)", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
<img width="723" height="452" alt="image" src="https://github.com/user-attachments/assets/d9a8f588-388c-41ce-b6e5-2f5e5e2fef17" />

## پیش‌بینی متن جدید
می‌توانیم یک متن دلخواه وارد کنیم و مدل پیش‌بینی کند که به کدام کلاس تعلق دارد.
همچنین احتمال تعلق متن به هر کلاس نمایش داده می‌شود.
```
new_text = ["The stock market reacted positively to the new economic policy"]
new_vec = vectorizer.transform(new_text)
prediction = model.predict(new_vec)
proba = model.predict_proba(new_vec)

print("Custom news prediction:", prediction[0])
print("Prediction probabilities:", proba)
```
# output:

```
World: 0.08%
Sports: 0.00%
Business: 99.89%
Sci/Tech: 0.03%
```
متن جدید با احتمالات بالا به کلاس ها تعلق دارد. باتوجه به اعداد، خبر جدید مربوط به Business است.

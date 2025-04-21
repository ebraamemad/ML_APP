import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

# 1. تحميل البيانات
data_path = Path('E:\\projects\\student-mat.csv')
df = pd.read_csv(data_path, delimiter=';')

# 2. معالجة البيانات
# مثال: تحويل المتغيرات الفئوية
df = pd.get_dummies(df, columns=['school', 'sex', 'address', 'famsize'])

# 3. تحديد الميزات والهدف
X = df.drop('G3', axis=1)  # G3 هي الدرجة النهائية (الهدف)
y = df['G3']

# 4. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب النموذج
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. التقييم
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# 7. حفظ النموذج
model_path = Path("models/student_performance_model.pkl")
model_path.parent.mkdir(exist_ok=True)
joblib.dump(model, model_path)
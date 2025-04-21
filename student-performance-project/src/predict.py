import joblib
import pandas as pd
from pathlib import Path

# تحميل النموذج
model = joblib.load(Path("models/student_performance_model.pkl"))

# بيانات جديدة للتنبؤ (مثال)
new_data = pd.DataFrame({
    'age': [18],
    'school': [1],
    'address': [0],
    'traveltime': [2],
    'studytime': [3],
    'failures': [0],
    'school_GP': [1],
    'school_MS': [0],
    'sex_F': [1],
    'sex_M': [0]
})

# التنبؤ
prediction = model.predict(new_data)
print(f"الدرجة المتوقعة: {prediction[0]:.1f}")
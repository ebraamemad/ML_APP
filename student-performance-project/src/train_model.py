import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

df=pd.read_csv("E:\projects\ML_APP\student-performance-project\data\processed\initial_load.csv")
# Prepare data for modeling
# Exclude G1 and G2 as mentioned in the dataset description
X = df.drop(['G1', 'G2', 'G3'], axis=1)
y = df['G3']

# Encode categorical variables
le = LabelEncoder()
for column in X.select_dtypes(include=['object']):
    X[column] = le.fit_transform(X[column])

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
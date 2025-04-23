from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# تحميل النموذج والبيانات المطلوبة
model = joblib.load("student_performance_model.pkl")  # تأكد من وجود الملف
df = pd.read_csv("student-mat.csv", delimiter=";")

# تعريف هيكل بيانات الإدخال
class StudentData(BaseModel):
    school: str
    sex: str
    age: int
    address: str 
    famsize: str 
    Parrent_status: str
    Mother_edu: str
    Father_edu: str


app = FastAPI()

@app.post("/predict")
async def predict(data: StudentData):
    try:
        # تحويل البيانات إلى DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # ترميز المتغيرات الفئوية بنفس طريقة التدريب
        categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                          'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                          'famsup', 'paid', 'activities', 'nursery', 'higher',
                          'internet', 'romantic']
        
        le = LabelEncoder()
        for col in categorical_cols:
            input_data[col] = le.fit_transform(input_data[col])
        
        # التنبؤ
        prediction = model.predict(input_data)
        
        return {"predicted_grade": round(float(prediction[0]), 2)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import pandas as pd
import os
from pathlib import Path

def load_raw_data():
    """تحميل البيانات الخام من ملف CSV"""
    # تحديد مسار الملف
    data_path = Path("data/raw/Student_Performance.csv")  # تأكد من اسم الملف الفعلي
    
    # قراءة البيانات
    df = pd.read_csv(data_path)
    
    # طباعة معلومات أساسية للتأكد من التحميل الصحيح
    print("تم تحميل البيانات بنجاح!")
    print(f"عدد الصفوف والأعمدة: {df.shape}")
    print("\nعينة من البيانات:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # إنشاء المجلدات إذا لم تكن موجودة
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # تحميل البيانات
    student_data = load_raw_data()
    
    # حفظ نسخة أولية في المجلد processed (اختياري)
    student_data.to_csv("data/processed/initial_load.csv", index=False)
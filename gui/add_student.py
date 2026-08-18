import streamlit as st
import cv2
import numpy as np
import os
from database import add_student, add_student_image
from core.face_engine import FaceEngine

# تهيئة الـ FaceEngine مرة واحدة
@st.cache_resource
def load_engine():
    return FaceEngine()

def render_add_student_page():
    st.header("إضافة طالب جديد (Register Student)")
    
    face_engine = load_engine()
    
    # 1. إدخال اسم الطالب
    student_name = st.text_input("اسم الطالب بالكامل:")
    
    # 2. اختيار طريقة رفع الصور
    input_method = st.radio("طريقة إدخال الصور (مطلوب 5 صور):", ["رفع 5 صور من الجهاز", "التقاط 5 صور بالكاميرا"])
    
    images_to_process = []
    
    if input_method == "رفع 5 صور من الجهاز":
        uploaded_files = st.file_uploader("اختر 5 صور للطالب", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded_files:
            if len(uploaded_files) != 5:
                st.warning(f"رجاءً اختر 5 صور بالضبط. العدد الحالي: {len(uploaded_files)}")
            else:
                for file in uploaded_files:
                    bytes_data = file.read()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    images_to_process.append((file.name, cv_img))
                    
    elif input_method == "التقاط 5 صور بالكاميرا":
        st.info("قم بالتقاط 5 صور متتالية للطالب مع تغيير زاوية الوجه قليلاً في كل صورة.")
        for i in range(5):
            picture = st.camera_input(f"التقط الصورة رقم {i+1}", key=f"cam_{i}")
            if picture:
                bytes_data = picture.read()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                images_to_process.append((f"cam_img_{i+1}.jpg", cv_img))

    # 3. حفظ البيانات واستخراج الـ Embeddings
    if st.button("حفظ الطالب والوجوه في الداتابيز"):
        if not student_name.strip():
            st.error("يرجى إدخال اسم الطالب أولاً.")
            return
            
        if len(images_to_process) != 5:
            st.error("يجب توفير 5 صور بالضبط لاستكمال التسجيل.")
            return

        # حفظ الطالب أولاً في جدول Students
        stu_id = add_student(student_name)
        
        # إنشاء مجلد لحفظ صور الطالب
        student_dir = os.path.join("data", "student_images", f"stu_{stu_id}")
        os.makedirs(student_dir, exist_ok=True)
        
        saved_count = 0
        
        with st.spinner("جاري استخراج الـ Embeddings وحفظ الصور..."):
            for idx, (filename, img) in enumerate(images_to_process):
                # استخراج الـ Embedding والوجه المقصوص
                embedding, cropped_face = face_engine.get_face_embedding(img)
                
                if embedding is None:
                    st.warning(f"لم يتم الكشف عن وجه في الصورة رقم {idx+1} ({filename}) - تم تخطيها.")
                    continue
                
                # حفظ الصورة على الهارد
                save_path = os.path.join(student_dir, f"img_{idx+1}.jpg")
                cv2.imwrite(save_path, img)
                
                # حفظ المسار والـ Embedding في جدول Student_Images
                add_student_image(stu_id=stu_id, image_path=save_path, embedding_array=embedding)
                saved_count += 1

        if saved_count > 0:
            st.success(f"تم تسجيل الطالب '{student_name}' بنجاح! تم حفظ {saved_count} صور والـ Embeddings الخاصة بها.")
        else:
            st.error("فشل حفظ الطالب لأنه لم يتم الكشف عن أوجُه في الصور المرفوعة.")
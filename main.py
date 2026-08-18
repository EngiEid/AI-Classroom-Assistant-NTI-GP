import streamlit as st
from streamlit_option_menu import option_menu
from database import init_db, add_subject, add_class, get_connection
from gui.add_student import render_add_student_page
from gui.dashboard import render_dashboard_page
from core.video_processor import VideoProcessor
import os
import pandas as pd

# 1. تهيئة قاعدة البيانات عند تشغيل التطبيق
init_db()

st.set_page_config(
    page_title="Smart Classroom System",
    page_icon="🎓",
    layout="wide"
)

# 2. القائمة الجانبية (Sidebar Navigation)
with st.sidebar:
    st.title("🎓 نظام الفصول الذكية")
    selected = option_menu(
        menu_title="القائمة الرئيسية",
        options=["تسجيل طالب جديد", "معالجة فيديو المحاضرة", "لوحة التحليلات (Dashboard)", "إدارة المواد والصفوف"],
        icons=["person-plus", "camera-video", "bar-chart-line", "book"],
        default_index=0,
    )

# --- الصفحة الأولى: إضافة طالب ---
if selected == "تسجيل طالب جديد":
    render_add_student_page()

# --- الصفحة الثانية: رفع ومعالجة الفيديو ---
elif selected == "معالجة فيديو المحاضرة":
    st.header("📹 رفع ومعالجة فيديو المحاضرة")
    
    conn = get_connection()
    classes_df = pd.read_sql_query("""
        SELECT c.Class_ID, s.Name as Subject_Name 
        FROM Classes c 
        JOIN Subjects s ON c.Sub_ID = s.Sub_ID
    """, conn)
    conn.close()

    if classes_df.empty:
        st.error("يرجى إنشاء مادة وكلاس أولاً من صفحة 'إدارة المواد والصفوف'.")
    else:
        class_options = {f"Class #{row['Class_ID']} - {row['Subject_Name']}": row['Class_ID'] 
                         for _, row in classes_df.iterrows()}
        selected_class_name = st.selectbox("اختر المحاضرة المرتبطة بالفيديو:", list(class_options.keys()))
        selected_class_id = class_options[selected_class_name]

        uploaded_video = st.file_uploader("قم برفع فيديو المحاضرة", type=['mp4', 'avi', 'mov'])

        if uploaded_video is not None:
            # حفظ الفيديو مؤقتاً
            temp_video_path = os.path.join("data", "temp_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())

            st.video(temp_video_path)

            if st.button("بدء تحليل الفيديو وتتبع الطلاب"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"جاري معالجة الفيديو... {int(progress * 100)}%")

                # استدعاء المعالج
                processor = VideoProcessor()
                success = processor.process_video_file(
                    video_path=temp_video_path,
                    class_id=selected_class_id,
                    progress_callback=update_progress
                )

                if success:
                    st.success("تمت معالجة الفيديو بنجاح وتسجيل الحضور والتصرفات!")

# --- الصفحة الثالثة: الـ Dashboard ---
elif selected == "لوحة التحليلات (Dashboard)":
    render_dashboard_page()

# --- الصفحة الرابعة: إضافة المواد والكلاسات ---
elif selected == "إدارة المواد والصفوف":
    st.header("📚 إدارة المواد والمحاضرات")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("إضافة مادة جديدة")
        sub_name = st.text_input("اسم المادة:")
        if st.button("حفظ المادة"):
            if sub_name:
                sub_id = add_subject(sub_name)
                st.success(f"تمت إضافة المادة '{sub_name}' برقم ID: {sub_id}")
            else:
                st.error("يرجى إدخال اسم المادة.")

    with col2:
        st.subheader("إنشاء محاضرة/كلاس لمادة")
        conn = get_connection()
        subs_df = pd.read_sql_query("SELECT * FROM Subjects", conn)
        conn.close()

        if not subs_df.empty:
            sub_options = {row['Name']: row['Sub_ID'] for _, row in subs_df.iterrows()}
            selected_sub = st.selectbox("اختر المادة:", list(sub_options.keys()))
            
            if st.button("إنشاء كلاس جديد"):
                class_id = add_class(sub_options[selected_sub])
                st.success(f"تم إنشاء كلاس جديد للمادة برقم Class ID: {class_id}")
        else:
            st.info("قم بإضافة مادة أولاً للتمكن من إنشاء كلاسات.")
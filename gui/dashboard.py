import streamlit as st
import pandas as pd
import plotly.express as px
from database import get_connection

def render_dashboard_page():
    st.header("📊 لوحة التحليلات والتقارير (Dashboard)")

    conn = get_connection()

    # --- 1. اختيار الكلاس / المحاضرة ---
    classes_df = pd.read_sql_query("""
        SELECT c.Class_ID, s.Name as Subject_Name 
        FROM Classes c 
        JOIN Subjects s ON c.Sub_ID = s.Sub_ID
    """, conn)

    if classes_df.empty:
        st.warning("لا توجد محاضرات أو كلاسات مسجلة في النظام حتى الآن.")
        conn.close()
        return

    class_options = {f"Class #{row['Class_ID']} - {row['Subject_Name']}": row['Class_ID'] 
                     for _, row in classes_df.iterrows()}
    
    selected_class_name = st.selectbox("اختر المحاضرة لعرض التقرير:", list(class_options.keys()))
    selected_class_id = class_options[selected_class_name]

    st.markdown("---")

    # --- 2. عرض ملخص الحضور والغياب ---
    st.subheader("📌 تقرير الحضور (Attendance Summary)")

    attendance_df = pd.read_sql_query("""
        SELECT s.Stu_ID, s.Name as Student_Name, a.Status, a.Timestamp, a.Confidence
        FROM Attendance a
        JOIN Students s ON a.Stu_ID = s.Stu_ID
        WHERE a.Class_ID = ?
    """, conn, params=(selected_class_id,))

    # جلب جميع الطلاب المسجلين لحساب الغياب
    all_students_df = pd.read_sql_query("SELECT Stu_ID, Name FROM Students", conn)

    if not all_students_df.empty:
        attended_ids = attendance_df['Stu_ID'].tolist() if not attendance_df.empty else []
        all_students_df['Status'] = all_students_df['Stu_ID'].apply(lambda x: 'Present' if x in attended_ids else 'Absent')

        col1, col2, col3 = st.columns(3)
        total_students = len(all_students_df)
        present_count = len(attended_ids)
        absent_count = total_students - present_count

        col1.metric("إجمالي الطلاب", total_students)
        col2.metric("الحاضرون", present_count)
        col3.metric("الغائبون", absent_count)

        # رسم بياني للحضور
        fig_att = px.pie(all_students_df, names='Status', title="نسبة الحضور والغياب", 
                         color='Status', color_discrete_map={'Present':'#2ecc71', 'Absent':'#e74c3c'})
        st.plotly_chart(fig_att, use_container_width=True)

        st.dataframe(all_students_df[['Stu_ID', 'Name', 'Status']], use_container_width=True)

    # --- 3. عرض تحليلات الأحداث والتصرفات ---
    st.markdown("---")
    st.subheader("🎭 تحليل تصرفات الطلاب (Student Events)")

    events_df = pd.read_sql_query("""
        SELECT e.Event_ID, s.Name as Student_Name, e.Event_Type, e."Start Time", e."End Time", e.Confidence
        FROM Student_Events e
        JOIN Students s ON e.Stu_ID = s.Stu_ID
        WHERE e.Class_ID = ?
    """, conn, params=(selected_class_id,))

    if events_df.empty:
        st.info("لم يتم تسجيل أي أحداث أو تصرفات لهذه المحاضرة.")
    else:
        # رسم بياني لتوزيع الأحداث
        event_counts = events_df['Event_Type'].value_counts().reset_index()
        event_counts.columns = ['Event_Type', 'Count']
        
        fig_events = px.bar(event_counts, x='Event_Type', y='Count', 
                            title="توزيع تصرفات الطلاب خلال المحاضرة",
                            labels={'Event_Type': 'نوع التصرف', 'Count': 'التكرار'},
                            color='Event_Type')
        st.plotly_chart(fig_events, use_container_width=True)

        # جدول التفاصيل
        st.write("**تفاصيل الأحداث المسجلة:**")
        st.dataframe(events_df, use_container_width=True)

    conn.close()
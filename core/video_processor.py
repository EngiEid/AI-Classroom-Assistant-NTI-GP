import cv2
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
from core.face_engine import FaceEngine
from database import get_all_known_faces, record_attendance, record_event

class VideoProcessor:
    def __init__(self, behavior_model_path="yolov8n.pt", sim_threshold=0.5):
        """
        behavior_model_path: مسار أوزان موديل الـ Object Detection المتخصص في التصرفات
        """
        # 1. تحميل موديل الـ Detection والـ Tracking
        self.model = YOLO(behavior_model_path)
        
        # 2. تهيئة الـ Face Engine وجلب الوجوه المسجلة من الداتابيز
        self.face_engine = FaceEngine(sim_threshold=sim_threshold)
        self.known_students = get_all_known_faces()
        
        # Q-Dict لربط Track_ID الخاص بـ YOLO مع Stu_ID الخاص بالداتابيز
        # { track_id: {"stu_id": stu_id, "confidence": conf} }
        self.tracked_students = {}
        
        # متابعة الأحداث النشطة لحساب Start Time و End Time
        # { (track_id, event_type): start_time_str }
        self.active_events = {}

    def _frame_to_timestamp(self, frame_idx, fps):
        """تحويل رقم الفريم إلى تنسيق زمن HH:MM:SS"""
        seconds = int(frame_idx / fps)
        return str(timedelta(seconds=seconds))

    def process_video_file(self, video_path, class_id, progress_callback=None):
        """
        معالجة الفيديو كاملاً، تسجيل الحضور، ورصد التصرفات
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        attended_students = set()  # لمنع تكرار تسجيل الحضور لنفس الطالب

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_timestamp = self._frame_to_timestamp(frame_idx, fps)
            
            # --- أ) تنفيذ Object Tracking مع YOLOv8 ---
            # الموديل سيعيد الـ Bounding Boxes والـ Track IDs والـ Classes (التصرفات)
            results = self.model.track(frame, persist=True, verbose=False)[0]
            
            current_frame_active_events = set()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()

                for bbox, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confidences):
                    event_type = self.model.names[cls_id]  # اسم التصرف (مثلاً: raising_hand, using_phone)
                    
                    # --- ب) التعرف على الوجه وربطه بـ Track_ID (إذا لم يكن معروفاً مسبقاً) ---
                    if track_id not in self.tracked_students:
                        x1, y1, x2, y2 = map(int, bbox)
                        # قص منطقة الشخص للكشف عن وجهه
                        person_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        
                        if person_crop.size > 0:
                            face_results = self.face_engine.process_frame(person_crop, self.known_students)
                            for face in face_results:
                                if face["stu_id"] is not None:
                                    self.tracked_students[track_id] = {
                                        "stu_id": face["stu_id"],
                                        "confidence": face["confidence"]
                                    }
                                    break
                    
                    # إذا تم التعرف على الطالب بنجاح
                    if track_id in self.tracked_students:
                        stu_id = self.tracked_students[track_id]["stu_id"]
                        stu_conf = self.tracked_students[track_id]["confidence"]
                        
                        # --- جـ) تسجيل الحضور (مرة واحدة فقط لكل طالب) ---
                        if stu_id not in attended_students:
                            record_attendance(
                                stu_id=stu_id,
                                class_id=class_id,
                                status="Present",
                                timestamp=current_timestamp,
                                confidence=float(stu_conf)
                            )
                            attended_students.add(stu_id)

                        # --- د) إدارة الأحداث والتصرفات (Student Events) ---
                        # استثناء كلاس الشخص العادي (person) والتركيز على التصرفات فقط
                        if event_type != "person":
                            event_key = (stu_id, event_type)
                            current_frame_active_events.add(event_key)

                            if event_key not in self.active_events:
                                # بداية حدث جديد
                                self.active_events[event_key] = {
                                    "start_time": current_timestamp,
                                    "confidence": float(conf)
                                }

            # --- هـ) إغلاق وتسجيل الأحداث التي انتهت في هذا الفريم ---
            ended_events = []
            for event_key, data in self.active_events.items():
                if event_key not in current_frame_active_events:
                    stu_id, event_type = event_key
                    record_event(
                        stu_id=stu_id,
                        class_id=class_id,
                        event_type=event_type,
                        start_time=data["start_time"],
                        end_time=current_timestamp,
                        confidence=data["confidence"]
                    )
                    ended_events.append(event_key)
            
            for event_key in ended_events:
                del self.active_events[event_key]

            # تحديث نسبة التقدم للـ GUI
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

        # إغلاق أي أحداث استمرت حتى نهاية الفيديو
        final_timestamp = self._frame_to_timestamp(frame_idx, fps)
        for (stu_id, event_type), data in self.active_events.items():
            record_event(
                stu_id=stu_id,
                class_id=class_id,
                event_type=event_type,
                start_time=data["start_time"],
                end_time=final_timestamp,
                confidence=data["confidence"]
            )

        cap.release()
        return True
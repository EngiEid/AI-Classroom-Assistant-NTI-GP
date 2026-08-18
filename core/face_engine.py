import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, sim_threshold=0.5):
        """
        تهيئة موديل InsightFace
        sim_threshold: الحد الأدنى للتطابق (Cosine Similarity) للتعرف على الطالب
        """
        self.sim_threshold = sim_threshold
        # تحميل الموديلات وتحديد الأجهزة المستخدمة (CPU/GPU)
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding(self, image):
        """
        استخراج الـ Embedding والوجه المقصوص لصورة واحدة
        Returns: (embedding_vector, cropped_face) أو (None, None)
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            return None, None

        faces = self.app.get(image)
        if len(faces) == 0:
            return None, None
        
        # أخذ أكبر وجه في الصورة من حيث المساحة
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # قص صورة الوجه
        bbox = face.bbox.astype(int)
        cropped_face = image[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
        
        return face.embedding, cropped_face

    def process_multiple_images(self, images_list):
        """
        معالجة قائمة تحتوي على 5 صور لطالب واحد
        Returns: قائمة من الـ Embeddings المقبولة وقائمة بالوجوه المقصوصة
        """
        valid_embeddings = []
        cropped_faces = []

        for img in images_list:
            emb, cropped = self.get_face_embedding(img)
            if emb is not None:
                valid_embeddings.append(emb)
                cropped_faces.append(cropped)

        return valid_embeddings, cropped_faces

    def get_average_embedding(self, embeddings_list):
        """
        حساب المتوسط الحسابي لعدة Embeddings لتكوين Vector فريد وممثل للطالب
        """
        if not embeddings_list:
            return None
        avg_emb = np.mean(embeddings_list, axis=0)
        # إعادة عمل Normalization للمتوسط ليكون طوله 1
        return avg_emb / np.linalg.norm(avg_emb)

    def compute_similarity(self, emb1, emb2):
        """حساب Cosine Similarity بين 2 Embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def recognize_face(self, current_embedding, known_students):
        """
        مقارنة الـ Embedding الحالي بكافة الـ Embeddings المخزنة في الداتابيز
        known_students: [{"stu_id": id, "embedding": array}, ...]
        Returns: (stu_id, confidence)
        """
        best_match_id = None
        max_similarity = -1.0

        for student in known_students:
            sim = self.compute_similarity(current_embedding, student["embedding"])
            if sim > max_similarity:
                max_similarity = sim
                best_match_id = student["stu_id"]

        if max_similarity >= self.sim_threshold:
            return best_match_id, float(max_similarity)
        
        return None, float(max_similarity)

    def process_frame(self, frame, known_students):
        """
        معالجة فريم فيديو واستخراج كافة الوجوه المكتشفة مع التعرف عليها
        """
        faces = self.app.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            stu_id, confidence = self.recognize_face(face.embedding, known_students)

            results.append({
                "bbox": bbox,
                "stu_id": stu_id,
                "confidence": confidence,
                "embedding": face.embedding
            })

        return results
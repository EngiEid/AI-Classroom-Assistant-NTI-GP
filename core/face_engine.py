import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, sim_threshold=0.5, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        self.sim_threshold = sim_threshold
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None, None

        try:
            faces = self.app.get(image)
        except Exception as e:
            print(f"[FaceEngine Error] Extraction failed: {e}")
            return None, None

        if not faces:
            return None, None
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        bbox = face.bbox.astype(int)
        h, w, _ = image.shape
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        cropped_face = image[y1:y2, x1:x2]
        
        return face.embedding, cropped_face

    def process_multiple_images(self, images_list):
        valid_embeddings = []
        cropped_faces = []

        for img in images_list:
            emb, cropped = self.get_face_embedding(img)
            if emb is not None:
                valid_embeddings.append(emb)
                cropped_faces.append(cropped)

        return valid_embeddings, cropped_faces

    def get_average_embedding(self, embeddings_list):
        if not embeddings_list:
            return None
        avg_emb = np.mean(embeddings_list, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm == 0:
            return avg_emb
        return avg_emb / norm

    def prepare_known_faces(self, known_students):
        if not known_students:
            return [], np.array([])

        known_ids = [s["stu_id"] for s in known_students]
        raw_matrix = np.array([s["embedding"] for s in known_students])
        
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        known_matrix = raw_matrix / norms
        
        return known_ids, known_matrix

    def recognize_face_vectorized(self, current_embedding, known_ids, known_embeddings_matrix):
        if known_embeddings_matrix.size == 0 or current_embedding is None:
            return None, 0.0

        curr_norm_val = np.linalg.norm(current_embedding)
        if curr_norm_val == 0:
            return None, 0.0

        curr_norm = current_embedding / curr_norm_val
        similarities = np.dot(known_embeddings_matrix, curr_norm)
        
        best_idx = np.argmax(similarities)
        max_similarity = float(similarities[best_idx])

        if max_similarity >= self.sim_threshold:
            return known_ids[best_idx], max_similarity
        
        return None, max_similarity

    def process_frame(self, frame, known_ids, known_matrix):
        if frame is None or frame.size == 0:
            return []

        faces = self.app.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            
            if known_matrix.size > 0:
                stu_id, confidence = self.recognize_face_vectorized(face.embedding, known_ids, known_matrix)
            else:
                stu_id, confidence = None, 0.0

            results.append({
                "bbox": bbox,
                "stu_id": stu_id,
                "confidence": confidence,
                "embedding": face.embedding
            })

        return results
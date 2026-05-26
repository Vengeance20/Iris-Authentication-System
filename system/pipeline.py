import cv2
import numpy as np
import torch
from core.preprocessing import detect_pupil_hough, detect_iris_radius, remove_eyelids_linear_hough, daugman_rubber_sheet
from core.antispoof import AntiSpoofPredictor
from core.authentication import IrisAuthenticator
from database.vector_db import VectorDatabase

class IrisBiometricSystem:
    def __init__(self, antispoof_path, auth_path, threshold=0.72):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Hệ thống khởi động trên thiết bị: {self.device}")
        
        self.antispoof = AntiSpoofPredictor(antispoof_path, self.device)
        self.auth = IrisAuthenticator(auth_path, self.device)
        self.db = VectorDatabase()
        self.threshold = threshold

    def _run_preprocessing(self, raw_image):
        if len(raw_image.shape) == 3:
            img_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = raw_image.copy()
            
        pupil_params = detect_pupil_hough(img_gray)
        iris_r = detect_iris_radius(img_gray, pupil_params[0], pupil_params[1], pupil_params[2])
        iris_params = (pupil_params[0], pupil_params[1], iris_r)
        
        valid_mask, _ = remove_eyelids_linear_hough(img_gray, pupil_params, iris_params)
        clean_iris = cv2.bitwise_and(img_gray, img_gray, mask=valid_mask)
        
        normalized = daugman_rubber_sheet(clean_iris, pupil_params, iris_params, width=512, height=64)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(normalized)

    def enroll_new_user(self, user_id, image_path):
        print(f"\n📥 [ĐĂNG KÝ] Đang xử lý cho User ID: {user_id}")
        if self.db.exists(user_id):
            return False, "❌ Thất bại: User ID đã tồn tại trên hệ thống."
            
        raw_img = cv2.imread(image_path)
        if raw_img is None: return False, "❌ Thất bại: Không đọc được file ảnh."
        
        if not self.antispoof.predict_is_real(raw_img):
            return False, "⚠️ TỪ CHỐI: Phát hiện mống mắt giả lập (PAD)!"
            
        try:
            enhanced_strip = self._run_preprocessing(raw_img)
            vector = self.auth.extract_feature_vector(enhanced_strip)
            self.db.save_user(user_id, vector)
            return True, f"✅ Thành công: Đã đăng ký mống mắt cho [{user_id}]."
        except Exception as e:
            return False, f"❌ Lỗi xử lý hình học: {str(e)}"

    def authenticate_user(self, user_id, image_path):
        print(f"\n🔑 [XÁC THỰC] Yêu cầu quét truy cập từ User ID: {user_id}")
        if not self.db.exists(user_id):
            return False, "❌ Từ chối: ID người dùng chưa đăng ký trên hệ thống."
            
        raw_img = cv2.imread(image_path)
        if raw_img is None: return False, "❌ Thất bại: Không đọc được file ảnh quét."
        
        if not self.antispoof.predict_is_real(raw_img):
            return False, "⚠️ NGUY HIỂM: Phát hiện tấn công giả mạo (Spoofing Attack)!"
            
        try:
            enhanced_strip = self._run_preprocessing(raw_img)
            live_vector = self.auth.extract_feature_vector(enhanced_strip)
            
            enrolled_vector = self.db.get_user_vector(user_id)
            score = np.dot(live_vector, enrolled_vector) / (np.linalg.norm(live_vector) * np.linalg.norm(enrolled_vector))
            print(f"   -> Điểm tương đồng: {score:.4f} (Ngưỡng an toàn: {self.threshold})")
            
            if score >= self.threshold:
                return True, f"✅ TRUY CẬP THÀNH CÔNG: Chào mừng [{user_id}]!"
            else:
                return False, "❌ TỪ CHỐI TRUY CẬP: Mống mắt không khớp với cơ sở dữ liệu gốc!"
        except Exception as e:
            return False, f"❌ Lỗi hệ thống bất ngờ: {str(e)}"
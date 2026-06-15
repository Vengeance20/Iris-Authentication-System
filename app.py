import streamlit as st
import os
import torchvision.models as models
import torch.nn as nn
from PIL import Image

# Import class hệ thống từ file main.py của bạn
from system.pipeline import IrisBiometricSystem

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ thống Sinh trắc học Mống mắt", page_icon="👁️", layout="centered")
st.title("👁️ Iris Authentication System")
st.markdown("Hệ thống nhận diện mống mắt tích hợp AI Chống giả mạo (Anti-Spoofing)")

# Tạo thư mục tạm để lưu ảnh upload từ Web
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. KHỞI TẠO VÀ CACHE MÔ HÌNH (Chỉ load 1 lần)
# ==========================================
@st.cache_resource
def load_system():
    # Khung mạng Anti-spoofing
    my_antispoof_backbone = models.resnet18()
    my_antispoof_backbone.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(my_antispoof_backbone.fc.in_features, 1) 
    )
    
# Khởi tạo hệ thống
    system = IrisBiometricSystem(
        antispoof_path="models/pad_model.pth",         
        auth_path="models/auth_model.pth",        
        threshold=0.72                    
    )
    return system

# Load hệ thống vào memory
try:
    iris_system = load_system()
except Exception as e:
    st.error(f"Lỗi khởi tạo mô hình: Kiểm tra lại file PAD.pth và authentication.pth. Chi tiết: {e}")
    st.stop()

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
# Tạo 2 tab (Đăng ký và Xác thực)
tab1, tab2 = st.tabs(["📥 Đăng ký (Enrollment)", "🔑 Xác thực (Authentication)"])

# --- TAB 1: ĐĂNG KÝ ---
with tab1:
    st.header("Đăng ký Mống mắt mới")
    enroll_user_id = st.text_input("Nhập User ID (VD: U001, MinhBui):", key="enroll_id")
    enroll_img = st.file_uploader("Tải lên ảnh mống mắt để đăng ký", type=["jpg", "jpeg", "png"], key="enroll_img")
    
    if enroll_img is not None:
        st.image(enroll_img, caption="Ảnh thu nhận", width=300)
        
    if st.button("Tiến hành Đăng ký", type="primary"):
        if not enroll_user_id or enroll_img is None:
            st.warning("Vui lòng nhập đầy đủ User ID và tải ảnh lên.")
        else:
            with st.spinner("Đang xử lý phân tích sinh trắc..."):
                # Lưu file tạm để đưa vào hàm của bạn
                temp_path = os.path.join(TEMP_DIR, enroll_img.name)
                with open(temp_path, "wb") as f:
                    f.write(enroll_img.getbuffer())
                
                # Chạy luồng Enroll
                success, msg = iris_system.enroll_new_user(enroll_user_id, temp_path)
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# --- TAB 2: XÁC THỰC ---
with tab2:
    st.header("Kiểm tra Quyền truy cập")
    auth_user_id = st.text_input("Nhập User ID của bạn:", key="auth_id")
    auth_img = st.file_uploader("Tải lên ảnh quét mống mắt hiện tại", type=["jpg", "jpeg", "png"], key="auth_img")
    
    if auth_img is not None:
        st.image(auth_img, caption="Ảnh quét Live", width=300)
        
    if st.button("Yêu cầu Mở cửa", type="primary"):
        if not auth_user_id or auth_img is None:
            st.warning("Vui lòng nhập đầy đủ User ID và tải ảnh lên.")
        else:
            with st.spinner("Đang kiểm tra Liveness và so khớp cơ sở dữ liệu..."):
                # Lưu file tạm
                temp_path = os.path.join(TEMP_DIR, auth_img.name)
                with open(temp_path, "wb") as f:
                    f.write(auth_img.getbuffer())
                
                # Chạy luồng Auth
                success, msg = iris_system.authenticate_user(auth_user_id, temp_path)
                
                if success:
                    st.balloons() # Hiệu ứng bóng bay chúc mừng mở cửa
                    st.success(msg)
                else:
                    st.error(msg)
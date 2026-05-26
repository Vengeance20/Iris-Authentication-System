import os
from system.pipeline import IrisBiometricSystem

def main():
    # Khởi tạo Gate System điều phối
    system = IrisBiometricSystem(
        antispoof_path="models/pad_model.pth",
        auth_path="models/auth_model.pth",
        threshold=0.72
    )
    
    print("\n🚀 HỆ THỐNG XÁC THỰC MỐNG MẮT ĐÃ SẴN SÀNG!")
    
    while True:
        print("\n" + "="*40)
        print("     MENU ĐIỀU KHIỂN HỆ THỐNG")
        print("="*40)
        print("1. Đăng ký mống mắt mới (Enrollment)")
        print("2. Xác thực mống mắt vào cửa (Authentication)")
        print("3. Thoát hệ thống")
        print("="*40)
        
        choice = input("👉 Mời bạn chọn chức năng (1-3): ").strip()
        
        if choice == "1":
            user_id = input("📝 Nhập User ID mới: ").strip()
            img_path = input("🖼️ Nhập đường dẫn ảnh chụp mắt: ").strip()
            if not os.path.exists(img_path):
                print("❌ Lỗi: Đường dẫn file ảnh không tồn tại!")
                continue
            success, msg = system.enroll_new_user(user_id, img_path)
            print(msg)
            
        elif choice == "2":
            user_id = input("🔑 Nhập User ID cần kiểm tra: ").strip()
            img_path = input("🖼️ Nhập đường dẫn ảnh quét mắt live: ").strip()
            if not os.path.exists(img_path):
                print("❌ Lỗi: Đường dẫn file ảnh không tồn tại!")
                continue
            success, msg = system.authenticate_user(user_id, img_path)
            print(msg)
            
        elif choice == "3":
            print("👋 Đang tắt hệ thống an toàn. Tạm biệt!")
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ, vui lòng chọn lại từ 1 đến 3.")

if __name__ == "__main__":
    main()
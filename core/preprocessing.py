import cv2
import numpy as np

def detect_pupil_hough(image):
    """
    Tìm tâm và bán kính đồng tử bằng Hough Circle Transform.
    Returns: (px, py, pr)
    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=20, minRadius=20, maxRadius=90
    )

    if circles is None:
        h, w = image.shape
        return (w//2, h//2, 30) # Trả về giá trị mặc định nếu không tìm thấy

    circles = np.uint16(np.around(circles))
    best_circle = None
    min_intensity = 255

    for circle in circles[0, :]:
        cx, cy, r = circle
        mask = np.zeros_like(image)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean_intensity = cv2.mean(image, mask=mask)[0]

        if mean_intensity < min_intensity:
            min_intensity = mean_intensity
            best_circle = (cx, cy, r)

    return best_circle

def detect_iris_radius(image, pupil_x, pupil_y, pupil_r):
    """
    Tìm bán kính mống mắt bằng phân tích Gradient xuyên tâm (Radial Gradient).
    Returns: best_radius (int)
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
    blurred = cv2.medianBlur(smoothed, 5)

    min_radius = int(pupil_r * 2.0)
    max_radius = int(pupil_r * 4.5)

    best_radius = min_radius
    max_gradient = 0
    h, w = blurred.shape

    for r in range(min_radius, max_radius, 2):
        num_points = max(int(2 * np.pi * r), 32)
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

        x_coords = pupil_x + r * np.cos(theta)
        y_coords = pupil_y + r * np.sin(theta)
        valid = (x_coords >= 1) & (x_coords < w-1) & (y_coords >= 1) & (y_coords < h-1)

        if not np.any(valid): continue

        inner_r = max(1, r - 3)
        outer_r = min(min(h, w)//2, r + 3)

        x_in = np.clip((pupil_x + inner_r * np.cos(theta[valid])).astype(int), 0, w-1)
        y_in = np.clip((pupil_y + inner_r * np.sin(theta[valid])).astype(int), 0, h-1)
        x_out = np.clip((pupil_x + outer_r * np.cos(theta[valid])).astype(int), 0, w-1)
        y_out = np.clip((pupil_y + outer_r * np.sin(theta[valid])).astype(int), 0, h-1)

        grad = np.mean(np.abs(blurred[y_out, x_out].astype(float) - blurred[y_in, x_in].astype(float)))
        if grad > max_gradient:
            max_gradient = grad
            best_radius = r

    return best_radius

def remove_eyelids_linear_hough(image, pupil_params, iris_params):
    """
    Dò tìm viền mí mắt trên/dưới bằng Linear Hough Transform và cắt bỏ.
    Returns: (valid_mask, detected_lines_img)
    """
    px, py, pr = pupil_params
    ix, iy, ir = iris_params
    h, w = image.shape

    # Khởi tạo mask vành khăn cơ bản
    base_mask = np.zeros_like(image)
    cv2.circle(base_mask, (ix, iy), ir, 255, -1)
    cv2.circle(base_mask, (px, py), pr, 0, -1)

    isolated_iris = cv2.bitwise_and(image, image, mask=base_mask)

    # Tìm biên cạnh
    blurred = cv2.GaussianBlur(isolated_iris, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 80)

    # Chạy Linear Hough Transform
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=40, maxLineGap=15
    )

    valid_mask = base_mask.copy()
    detected_lines_img = cv2.cvtColor(isolated_iris, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính góc của đường thẳng để lọc đường ngang
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

            if angle < 30 or angle > 150:
                cv2.line(detected_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Cắt phần trên
                if (y1 + y2) / 2 < py:
                    cv2.rectangle(valid_mask, (0, 0), (w, max(y1, y2)), 0, -1)
                # Cắt phần dưới
                elif (y1 + y2) / 2 > py:
                    cv2.rectangle(valid_mask, (0, min(y1, y2)), (w, h), 0, -1)

    return valid_mask, detected_lines_img

def daugman_rubber_sheet(image, pupil_params, iris_params, width=512, height=64):
    """
    Thuật toán Daugman: Trải phẳng mống mắt hình vành khăn thành chữ nhật.
    Sử dụng Numpy Vectorization để đạt tốc độ tối đa.
    
    Parameters:
        image: Ảnh xám mống mắt đã được làm sạch (chỉ còn vân mống mắt)
        pupil_params: Tuple (px, py, pr)
        iris_params: Tuple (ix, iy, ir)
        width: Độ phân giải góc (Theta) - Mặc định 512
        height: Độ phân giải bán kính (Radius) - Mặc định 64
        
    Returns:
        normalized_iris: Ảnh chữ nhật kích thước (height, width)
    """
    px, py, pr = pupil_params
    ix, iy, ir = iris_params

    # Tạo lưới tọa độ (theta, r)
    theta = np.linspace(0, 2 * np.pi, width)
    r_proportions = np.linspace(0, 1, height)

    # Tính ranh giới đồng tử (inner boundary)
    pupil_x = px + pr * np.cos(theta)
    pupil_y = py + pr * np.sin(theta)

    # Tính ranh giới mống mắt (outer boundary)
    iris_x = ix + ir * np.cos(theta)
    iris_y = iy + ir * np.sin(theta)

    # Khởi tạo ma trận ánh xạ tọa độ
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Nội suy tuyến tính khoảng cách giữa 2 ranh giới (inner và outer)
    for i, r_prop in enumerate(r_proportions):
        map_x[i, :] = (1 - r_prop) * pupil_x + r_prop * iris_x
        map_y[i, :] = (1 - r_prop) * pupil_y + r_prop * iris_y

    # Ánh xạ pixel từ ảnh gốc sang dải hình chữ nhật
    normalized_iris = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC)
    
    return normalized_iris
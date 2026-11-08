import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 設置 Streamlit 頁面設定
st.set_page_config(page_title="AI 輔助互動式影像處理與人臉分析工具", layout="wide")

# 強制注入 CSS 讓圖片完美貼合容器並保持比例縮放
st.markdown(
    """
    <style>
    /* 強制撐開圖片外層容器 */
    div[data-testid="stImage"] {
        width: 100% !important;
    }
    /* 強制圖片本體無條件隨容器縮放 (覆蓋任何行內樣式) */
    div[data-testid="stImage"] img {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖼️ 互動式影像處理與人臉分析工具 (期中專題)")
st.caption("基於 Streamlit + OpenCV + MediaPipe | AI 協作開發")

# =====================================================================
# 1. 影像載入與基礎解析 (Image Loading and Metadata)
# =====================================================================
st.sidebar.header("📥 步驟一：載入影像")
uploaded_file = st.sidebar.file_uploader("請選擇影像檔案", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # 讀取影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # 確保使用 OpenCV 讀取
    img_bgr = cv2.imdecode(file_bytes, 1)
    if img_bgr is None:
        st.error("影像讀取失敗，請確認檔案格式是否損毀。")
        st.stop()
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 獲取中繼資訊
    h, w, c = img_rgb.shape
    file_size_kb = uploaded_file.size / 1024

    # 在側邊欄顯示中繼資訊
    st.sidebar.subheader("ℹ️ 影像中繼資訊")
    st.sidebar.markdown(f"""
    - **檔名**: `{uploaded_file.name}`
    - **解析度 (寬 x 高)**: `{w} x {h}` 像素
    - **色彩維度 (Channels)**: `{c}` 通道 (RGB)
    - **檔案大小**: `{file_size_kb:.2f} KB`
    """)

    # =====================================================================
    # 2. 多色彩空間轉換與同步直方圖分析
    # =====================================================================
    st.header("🌈 色彩空間與直方圖分析")
    
    color_space_option = st.selectbox(
        "選擇檢視模式",
        ["原始影像 (Original)", 
         "R 通道 (Red Channel)", "G 通道 (Green Channel)", "B 通道 (Blue Channel)",
         "Y 亮度通道 (YUV-Y)", "U 色度通道 (YUV-U)", "V 色度通道 (YUV-V)"]
    )
    
    # 處理色彩空間轉換
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    
    display_img = None
    hist_title = ""
    hist_data = None
    color_map = "gray"
    
    if color_space_option == "原始影像 (Original)":
        display_img = img_rgb
        hist_title = "RGB 各通道強度直方圖"
    elif color_space_option == "R 通道 (Red Channel)":
        display_img = img_rgb[:, :, 0]
        hist_title = "紅色通道強度直方圖 (0-255)"
        hist_data = display_img
    elif color_space_option == "G 通道 (Green Channel)":
        display_img = img_rgb[:, :, 1]
        hist_title = "綠色通道強度直方圖 (0-255)"
        hist_data = display_img
    elif color_space_option == "B 通道 (Blue Channel)":
        display_img = img_rgb[:, :, 2]
        hist_title = "藍色通道強度直方圖 (0-255)"
        hist_data = display_img
    elif color_space_option == "Y 亮度通道 (YUV-Y)":
        display_img = img_yuv[:, :, 0]
        hist_title = "Y (亮度) 通道強度直方圖 (0-255)"
        hist_data = display_img
    elif color_space_option == "U 色度通道 (YUV-U)":
        display_img = img_yuv[:, :, 1]
        hist_title = "U (色度) 通道強度直方圖 (0-255)"
        hist_data = display_img
    elif color_space_option == "V 色度通道 (YUV-V)":
        display_img = img_yuv[:, :, 2]
        hist_title = "V (色度) 通道強度直方圖 (0-255)"
        hist_data = display_img

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ 影像預覽")
        if len(display_img.shape) == 3:
            # 加上 Image.fromarray() 強制轉換
            st.image(Image.fromarray(display_img), use_container_width=True)
        else:
            st.image(Image.fromarray(display_img), use_container_width=True)
            
    with col2:
        st.subheader("📊 同步直方圖")
        if color_space_option == "原始影像 (Original)":
            # 取得 RGB 三通道直方圖資料
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
            
            # 使用 Streamlit 內建折線圖
            st.line_chart(
                {"Red Channel": hist_r, "Green Channel": hist_g, "Blue Channel": hist_b},
                color=["#FF0000", "#00FF00", "#0000FF"],
                use_container_width=True
            )
        else:
            # 繪製單通道直方圖
            hist = cv2.calcHist([hist_data], [0], None, [256], [0, 256]).flatten()
            st.line_chart({f"{color_space_option} 強度": hist}, color=["#808080"], use_container_width=True)

    # =====================================================================
    # 3. 自訂卷積核 (Kernel Matrix) 空間濾波與量化評估
    # =====================================================================
    st.header("🔍 自訂卷積核與空間濾波")
    
    st.markdown("**PSNR 計算公式說明：**")
    st.latex(r"PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)")
    st.markdown(r"其中 $MAX_I = 255$。均方誤差 $MSE = \frac{1}{m \cdot n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$。PSNR 值越高，代表濾波影像與原圖的差異越小。")
    
    # 預設矩陣選項 (防呆：預設提供常用矩陣，以防使用者輸入錯誤)
    kernel_preset = st.selectbox(
        "選擇預設卷積核，或於下方自訂：",
        ["自訂輸入", "高斯平滑 (3x3)", "邊緣銳化 (3x3)", "浮雕效果 (3x3)", "拉普拉斯邊緣偵測 (3x3)"]
    )
    
    # 設定矩陣尺寸
    k_size = st.radio("矩陣尺寸", [3, 5], index=0, horizontal=True)
    
    # 初始化矩陣 values
    default_matrix = np.zeros((k_size, k_size))
    if k_size == 3:
        if kernel_preset == "高斯平滑 (3x3)":
            default_matrix = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        elif kernel_preset == "邊緣銳化 (3x3)":
            default_matrix = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        elif kernel_preset == "浮雕效果 (3x3)":
            default_matrix = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        elif kernel_preset == "拉普拉斯邊緣偵測 (3x3)":
            default_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        else:
            default_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif k_size == 5:
        default_matrix = np.zeros((5, 5))
        default_matrix[2, 2] = 1.0  # 單位矩陣作為預設

    # 顯示動態矩陣輸入 UI
    st.write("🔧 請輸入或編輯矩陣數值 (可用小數)：")
    inputs = []
    for r in range(k_size):
        cols_ui = st.columns(k_size)
        row_inputs = []
        for c_idx in range(k_size):
            val = cols_ui[c_idx].number_input(
                f"[{r},{c_idx}]", 
                value=float(default_matrix[r, c_idx]), 
                key=f"k_{r}_{c_idx}_{k_size}"
            )
            row_inputs.append(val)
        inputs.append(row_inputs)
        
    kernel = np.array(inputs, dtype=np.float32)
    
    # 正規化參數選擇
    normalize_kernel = st.checkbox("啟用正規化 (讓卷積核係數總和為 1)", value=False)
    
    # 防呆機制：防止 Kernel 全為 0 或和為 0 時除以零
    kernel_sum = np.sum(kernel)
    if normalize_kernel:
        if abs(kernel_sum) > 1e-6:
            kernel = kernel / kernel_sum
        else:
            st.warning("⚠️ 卷積核總和為 0，無法進行正規化，已自動取消正規化。")

    # 執行濾波 (套用在原始影像)
    # ddepth = -1 表示輸出影像深度與輸入一致
    filtered_img_bgr = cv2.filter2D(img_bgr, -1, kernel)
    filtered_img_rgb = cv2.cvtColor(filtered_img_bgr, cv2.COLOR_BGR2RGB)
    
    # 計算量化指標 PSNR
    # 防呆：確保輸入合法
    mse = np.mean((img_rgb.astype(np.float64) - filtered_img_rgb.astype(np.float64)) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 20 * np.log10(255.0 / np.sqrt(mse))

    # 顯示結果對比
    col_filter_1, col_filter_2 = st.columns(2)
    with col_filter_1:
        st.subheader("原始影像")
        st.image(Image.fromarray(img_rgb), use_container_width=True)
    with col_filter_2:
        st.subheader("濾波後影像")
        st.image(Image.fromarray(filtered_img_rgb), use_container_width=True)
        
    if psnr_val == float('inf'):
        st.success("📊 **量化評估：** 影像未改變，PSNR = ∞ dB")
    else:
        st.info(f"📊 **量化評估 (PSNR)：** `{psnr_val:.4f} dB` (與原圖差異度)")

    # =====================================================================
    # 4. 人臉偵測 (Face Detection) 模組 (改用 OpenCV Haar-Cascade)
    # =====================================================================
    st.header("🧑‍🤝‍🧑 人臉偵測 (Face Detection) 模組")
    
    # Haar-Cascade 的參數調整
    scale_factor = st.slider("縮放比例 (Scale Factor) - 控制多尺度偵測", 1.05, 1.50, 1.10, 0.05)
    min_neighbors = st.slider("最小相鄰矩形數 (Min Neighbors) - 數值越高標準越嚴格", 1, 10, 5, 1)
    
    if st.button("🚀 執行人臉偵測"):
        with st.spinner("人臉辨識中，請稍候..."):
            # 複製原圖以便標註
            annotated_img = img_rgb.copy()
            
            # Haar-Cascade 必須使用灰階影像進行偵測
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 載入 OpenCV 正臉偵測模型 (加入雲端自動下載防呆機制)
            import os
            import urllib.request
            
            cascade_path = "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                # 若雲端環境找不到內建檔案，自動從 OpenCV 官方下載
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, cascade_path)
                
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # 使用 detectMultiScale3 以取得權重 (當作 Confidence Score)
            faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(
                gray_img,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30),
                outputRejectLevels=True
            )
            
            face_count = len(faces) if len(faces) > 0 else 0
            
            if face_count > 0:
                for idx, (x, y, w_box, h_box) in enumerate(faces):
                    # 修正：直接取值並轉為 float，避免陣列維度錯誤
                    score = float(levelWeights[idx])
                    
                    # 繪製 Bounding Box
                    cv2.rectangle(annotated_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
                    
                    # 標示序號與分數
                    label = f"Face #{idx+1}: {score:.2f}"
                    cv2.putText(annotated_img, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.subheader("偵測結果影像")
            st.image(Image.fromarray(annotated_img), use_container_width=True)
            
            # 報告數據
            st.success(f"🔍 偵測完畢！共偵測到 **{face_count}** 張人臉。")
            if face_count > 0:
                for idx in range(face_count):
                    st.write(f"- 👦 人臉 #{idx+1}：推論分數 (Detection Weight/Score) = `{float(levelWeights[idx]):.4f}`")

else:
    # 初始未上傳影像時提示
    st.info("💡 請在左側邊欄上傳一個 PNG、JPG、JPEG 或 BMP 的影像檔案，即可展開影像與人臉分析之旅！")

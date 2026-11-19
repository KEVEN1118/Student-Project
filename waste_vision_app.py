"""Recycling Safety Vision Pro: High-Speed Dual-Mode Web Application.

Features an upgraded Industrial Command Center UI theme with 2x2 grid
intermediate OpenCV pipeline visualizations matching top container proportions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import math
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    torch = None

# ==========================================
# 1. Page Configuration & Custom UI CSS
# ==========================================
st.set_page_config(
    page_title="Recycling Safety Vision Pro",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Global App Background */
    .stApp {
        background-color: #0F172A;
    }

    /* Top Live Status Indicator Pulse */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Main Dashboard Header */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Industrial Metric Card Styling */
    .metric-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
    .card-total { border-top: 3px solid #3B82F6; }
    .card-high { border-top: 3px solid #EF4444; }
    .card-medium { border-top: 3px solid #F59E0B; }
    .card-speed { border-top: 3px solid #10B981; }

    /* Industrial Navigation Tabs Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E293B;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 600;
        padding: 0px 18px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #10B981 !important;
        color: #FFFFFF !important;
    }

    /* Sidebar Containers & Expanders */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stExpander {
        background-color: #1E293B;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Bar Layout
st.markdown(
    """
<div class="status-badge">
    <span class="status-pulse"></span> SYSTEM ONLINE — CONVEYOR BELT MONITOR ACTIVE
</div>
<div class="main-title">♻️ Recycling Safety Vision Pro</div>
<div class="sub-title">Automated Real-Time Hazardous Waste Detection and Material Classification System</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. Data Classes & Utility Functions
# ==========================================
@dataclass
class Detection:
    object_id: int
    detector_label: str
    estimated_category: str
    confidence: str
    review_priority: str
    reason: str
    x: int
    y: int
    width: int
    height: int
    area_px: int
    aspect_ratio: float
    solidity: float
    circularity: float


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_image(uploaded_file, max_width: int = 1200) -> np.ndarray:
    decoded = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("Selected file is not a supported image format.")

    h, w = decoded.shape[:2]
    if w > max_width:
        scale = max_width / w
        decoded = cv2.resize(
            decoded, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA
        )
    return decoded


def crop_roi_margin(image: np.ndarray, margin_percent: float) -> np.ndarray:
    if margin_percent <= 0:
        return image
    h, w = image.shape[:2]
    my = int(h * (margin_percent / 100.0))
    mx = int(w * (margin_percent / 100.0))
    if h - 2 * my <= 20 or w - 2 * mx <= 20:
        return image
    return image[my : h - my, mx : w - mx]


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def apply_clahe(bgr_img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)


def category_from_hsv_masked(
    hsv_roi: np.ndarray, mask: np.ndarray
) -> tuple[str, tuple[int, int, int], str]:
    if cv2.countNonZero(mask) == 0:
        return "Unknown", (120, 120, 120), "Empty Region"

    h, s, v, _ = cv2.mean(hsv_roi, mask=mask)
    h, s, v = round(h), round(s), round(v)
    evidence = f"Masked HSV: H={h}, S={s}, V={v}"

    if s <= 25 and v >= 205:
        return "Paper / Document Sheet", (50, 160, 230), evidence
    if s <= 45 and 160 <= v < 205:
        return "Plastic Bag / Soft Film", (220, 120, 220), evidence
    if 7 <= h <= 35 and 30 <= s <= 140 and 40 <= v <= 245:
        return "Paper / Cardboard", (42, 112, 165), evidence
    if 36 <= h <= 95 and s >= 45 and v >= 40:
        return "Organic / Food Waste", (35, 145, 35), evidence
    if s <= 50 and v >= 80:
        return "Metal or Clear Rigid Plastic", (180, 180, 180), evidence
    if s >= 70 and 85 <= h <= 140:
        return "Coloured Plastic", (220, 90, 30), evidence
    if s >= 65 and (h <= 10 or h >= 165):
        return "Printed Packaging / Can", (30, 30, 210), evidence

    return "Mixed or Uncertain Material", (120, 120, 120), evidence


def apply_watershed_decoupling(
    bgr: np.ndarray, binary_mask: np.ndarray
) -> np.ndarray:
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_val = dist_transform.max()
    if max_val == 0:
        return binary_mask

    _, sure_fg = cv2.threshold(dist_transform, 0.40 * max_val, 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    if num_labels <= 1:
        return binary_mask

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)

    separated_mask = np.zeros_like(binary_mask)
    separated_mask[markers > 1] = 255
    return separated_mask


# ==========================================
# 3. Core Processing Pipelines
# ==========================================
@st.cache_data(max_entries=5, show_spinner=False)
def classical_analysis(
    bgr: np.ndarray,
    auto_tune: bool = True,
    use_clahe: bool = True,
    use_watershed: bool = False,
    filter_type: str = "Bilateral Filter (Edge-Preserving)",
    kernel_size: int = 5,
    manual_canny_low: int = 25,
    manual_canny_high: int = 70,
    manual_min_area: int = 150,
):
    t0 = time.perf_counter()
    img_h, img_w = bgr.shape[:2]
    total_pixels = img_h * img_w

    if auto_tune:
        gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        canny_low, canny_high = auto_canny(gray_raw)
        minimum_area = int(total_pixels * 0.001)
    else:
        canny_low = manual_canny_low
        canny_high = manual_canny_high
        minimum_area = manual_min_area

    processed_bgr = apply_clahe(bgr) if use_clahe else bgr.copy()
    gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)

    if filter_type == "Bilateral Filter (Edge-Preserving)":
        filtered = cv2.bilateralFilter(
            gray, d=kernel_size, sigmaColor=75, sigmaSpace=75
        )
    else:
        filtered = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges = cv2.Canny(filtered, canny_low, canny_high)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    binary_mask = cv2.dilate(binary_mask, morph_kernel, iterations=1)

    if use_watershed:
        final_mask = apply_watershed_decoupling(processed_bgr, binary_mask)
    else:
        final_mask = binary_mask

    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    hsv = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
    annotated = bgr.copy()
    detections: list[Detection] = []

    max_allowable_area = int(total_pixels * 0.80)
    min_red_alert_area = max(600, int(total_pixels * 0.003))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area > max_allowable_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w > int(img_w * 0.85) and h > int(img_h * 0.85):
            continue

        ratio = max(w, h) / max(1, min(w, h))
        if ratio > 8.0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        perimeter = cv2.arcLength(contour, True)
        circularity = (
            float(4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        )

        local_cnt = contour - np.array([[[x, y]]], dtype=contour.dtype)
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [local_cnt], -1, 255, thickness=cv2.FILLED)

        category, default_color, evidence = category_from_hsv_masked(
            hsv[y : y + h, x : x + w], contour_mask
        )

        is_large_enough_for_alert = area >= min_red_alert_area
        is_sharp_shape = (
            (ratio >= 2.5 and solidity < 0.60) or (solidity < 0.50 and circularity < 0.30)
        )

        if is_large_enough_for_alert and is_sharp_shape:
            priority = "HIGH"
            box_color = (0, 0, 255)
            display_category = "High Hazard Sharp/Strip"
            reason = "Irregular geometry & significant area"
        elif ratio >= 1.8:
            priority = "MEDIUM"
            box_color = default_color
            display_category = category
            reason = "Elongated waste object region"
        else:
            priority = "LOW"
            box_color = default_color
            display_category = category
            reason = "Standard segmented object region"

        detections.append(
            Detection(
                object_id=0,
                detector_label="contour region",
                estimated_category=display_category,
                confidence=evidence,
                review_priority=priority,
                reason=reason,
                x=x,
                y=y,
                width=w,
                height=h,
                area_px=int(area),
                aspect_ratio=round(ratio, 2),
                solidity=round(solidity, 3),
                circularity=round(circularity, 3),
            )
        )

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            annotated,
            f"#{len(detections)} {display_category[:16]} | {priority}",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            2,
            cv2.LINE_AA,
        )

    detections.sort(key=lambda d: (d.y, d.x))
    for idx, det in enumerate(detections, 1):
        det.object_id = idx

    exec_time_ms = (time.perf_counter() - t0) * 1000

    views = {
        "HSV Colour Space": cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
        "Filtered Image": filtered,
        "Canny Edges": edges,
        "Morphology Mask": final_mask,
        "Annotated Result": to_rgb(annotated),
    }
    return views, detections, exec_time_ms


def yolo_category(label: str) -> tuple[str, tuple[int, int, int]]:
    sharp_hazards = {"knife", "scissors", "fork", "syringe", "razor"}
    paper_classes = {"book", "newspaper", "paper", "cardboard"}
    plastic_bag_classes = {"handbag", "backpack", "suitcase", "bag", "plastic bag"}
    food_classes = {
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake",
    }

    if label in sharp_hazards:
        return "Hazardous Sharps / Danger", (0, 0, 255)

    if label in paper_classes:
        return "Paper / Cardboard Product", (165, 112, 42)
    if label in plastic_bag_classes:
        return "Plastic Bag / Packaging Film", (220, 120, 220)
    if label in food_classes:
        return "Organic / Food Waste", (35, 145, 35)

    category_map = {
        "bottle": ("Plastic / Glass Bottle", (30, 90, 220)),
        "cup": ("Cup / Container", (220, 180, 0)),
        "wine glass": ("Glass Container", (190, 190, 190)),
        "bowl": ("Packaging / Bowl", (200, 120, 180)),
        "can": ("Metal / Aluminium Can", (210, 30, 30)),
        "cell phone": ("E-Waste / Electronics", (200, 50, 200)),
        "laptop": ("E-Waste / Electronics", (200, 50, 200)),
    }
    return category_map.get(
        label, (f"{label.capitalize()} (Material Review)", (120, 120, 120))
    )


@st.cache_resource(show_spinner="Loading YOLOv8 AI Model Weights...")
def load_yolo(model_name: str):
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if YOLO is None:
        raise RuntimeError("Ultralytics package is not installed.")
    return YOLO(model_name)


@st.cache_data(max_entries=5, show_spinner=False)
def yolo_analysis(
    bgr: np.ndarray,
    model_name: str,
    threshold: float,
    iou_thresh: float,
    img_size: int,
):
    model = load_yolo(model_name)

    if torch and torch.cuda.is_available():
        device = "0"
    elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    t0 = time.perf_counter()
    result = model(
        bgr,
        conf=threshold,
        iou=iou_thresh,
        imgsz=img_size,
        device=device,
        half=(device != "cpu"),
        verbose=False,
    )[0]
    exec_time_ms = (time.perf_counter() - t0) * 1000

    annotated = bgr.copy()
    detections: list[Detection] = []
    sharp_hazards_set = {"knife", "scissors", "fork", "syringe", "razor"}

    for idx, box in enumerate(result.boxes, 1):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        score = float(box.conf[0])
        label = result.names[int(box.cls[0])]

        category, color = yolo_category(label)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        ratio = max(w, h) / min(w, h)

        priority = (
            "HIGH"
            if label in sharp_hazards_set
            else "MEDIUM"
            if label in {"bottle", "wine glass", "cup"}
            else "LOW"
        )
        reason = (
            f"Hazardous item ({label}) detected by YOLO"
            if priority == "HIGH"
            else f"Localized {label}"
        )

        detections.append(
            Detection(
                object_id=idx,
                detector_label=label,
                estimated_category=category,
                confidence=f"{score:.0%}",
                review_priority=priority,
                reason=reason,
                x=x1,
                y=y1,
                width=w,
                height=h,
                area_px=w * h,
                aspect_ratio=round(ratio, 2),
                solidity=1.0,
                circularity=1.0,
            )
        )

        box_color = (0, 0, 255) if priority == "HIGH" else color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            annotated,
            f"#{idx} {category[:16]} | {score:.0%}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return to_rgb(annotated), detections, exec_time_ms


def png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def render_metric_card(label: str, value: str | int, card_type: str = "total"):
    st.markdown(
        f"""
    <div class="metric-card card-{card_type}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ==========================================
# 4. Streamlit Application UI Layout
# ==========================================
with st.sidebar:
    st.header("⚙️ Algorithm Controls")

    auto_tune = st.checkbox("🤖 Enable Smart Auto-Tuning (No Manual Setup Required)", value=True)

    with st.expander("✂️ Conveyor Belt ROI Cropping", expanded=False):
        crop_margin = st.slider("Crop Belt Outer Margin (%)", 0, 20, 0, step=1)

    with st.expander("🔹 OpenCV Parameters", expanded=not auto_tune):
        use_clahe = st.checkbox("Enable CLAHE Lighting Equalizer", value=True)
        use_watershed = st.checkbox("Enable Watershed Decoupling (Experimental)", value=False)
        filter_type = st.selectbox(
            "Filter Type",
            ["Bilateral Filter (Edge-Preserving)", "Gaussian Blur"],
        )
        kernel_size = st.selectbox("Filter Kernel Size", [3, 5, 7, 9], index=1)
        manual_canny_low = st.slider("Canny Low Threshold", 0, 250, 25, disabled=auto_tune)
        manual_canny_high = st.slider("Canny High Threshold", 1, 300, 70, disabled=auto_tune)
        manual_min_area = st.slider("Min Area Threshold (px)", 50, 3000, 150, step=50, disabled=auto_tune)

    with st.expander("🤖 YOLOv8 AI Parameters", expanded=True):
        model_name = st.selectbox(
            "Model Weights",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
            index=0,
        )
        img_size = st.selectbox(
            "Inference Res (imgsz)", [640, 1280, 320], index=0
        )
        yolo_threshold = st.slider(
            "Confidence Threshold (Conf)", 0.05, 0.95, 0.25, 0.05
        )
        iou_threshold = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    if st.button("🧹 Clear Memory Cache & Reset RAM"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.success("Memory Cache & VRAM successfully cleared!")

uploaded = st.file_uploader(
    "📸 Upload Waste / Conveyor Belt Image Stream",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info(
        "💡 **Getting Started:** Upload a conveyor belt stream image to execute real-time feature extraction and neural localization."
    )
    st.stop()

if not auto_tune and manual_canny_high <= manual_canny_low:
    st.error("Canny High Threshold must be greater than Low Threshold.")
    st.stop()

try:
    raw_loaded = load_image(uploaded)
    original = crop_roi_margin(raw_loaded, crop_margin)
except ValueError as err:
    st.error(str(err))
    st.stop()

tab_opencv, tab_yolo, tab_benchmark = st.tabs([
    "🔹 OpenCV Classical Mode",
    "🤖 YOLOv8 Deep Learning",
    "⚡ Dual-Mode Benchmark",
])

# ---------------------------------------------------------
# Tab 1: OpenCV
# ---------------------------------------------------------
with tab_opencv:
    st.markdown("### 🔹 Classical OpenCV Segmentation")
    views, classical_dets, cv_latency = classical_analysis(
        original,
        auto_tune,
        use_clahe,
        use_watershed,
        filter_type,
        kernel_size,
        manual_canny_low,
        manual_canny_high,
        manual_min_area,
    )

    h_count = sum(d.review_priority == "HIGH" for d in classical_dets)
    m_count = sum(d.review_priority == "MEDIUM" for d in classical_dets)

    if h_count > 0:
        st.error(
            f"🚨 **【HIGH RISK HAZARD ALERT】 Detected {h_count} sharp hazard(s) / irregular strips on the conveyor belt! Objects highlighted in RED. Pause belt for manual review!**"
        )

    # Main Output Comparison (Top 2 Columns)
    c1, c2 = st.columns(2)
    c1.image(
        to_rgb(original),
        caption=f"Input Image: {uploaded.name}",
        use_container_width=True,
    )
    c2.image(
        views["Annotated Result"],
        caption="OpenCV Hazard Segmentation Output",
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Intermediate Pipeline Steps in 2x2 Grid (Matching Top Columns Scale)
    with st.expander("🔬 Intermediate Computer Vision Pipeline Processing Steps", expanded=True):
        p1, p2 = st.columns(2)
        p1.image(
            views["HSV Colour Space"],
            caption="1. HSV Colour Space",
            use_container_width=True,
        )
        p2.image(
            views["Filtered Image"],
            caption="2. Bilateral / Gaussian Filter",
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        p3, p4 = st.columns(2)
        p3.image(
            views["Canny Edges"],
            caption="3. Auto-Canny Edges",
            use_container_width=True,
        )
        p4.image(
            views["Morphology Mask"],
            caption="4. Morphological Binary Mask",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Contours", len(classical_dets), "total")
    with m2:
        render_metric_card("High Risk (Sharps/Strips)", h_count, "high")
    with m3:
        render_metric_card("Medium Cues", m_count, "medium")
    with m4:
        render_metric_card("Pipeline Latency", f"{cv_latency:.1f} ms", "speed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Classical CV Detailed Inspection Table")
    if classical_dets:
        df_cv = pd.DataFrame([asdict(d) for d in classical_dets])
        st.dataframe(df_cv, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button(
            "📥 Download OpenCV CSV Report",
            df_cv.to_csv(index=False).encode("utf-8"),
            "opencv_inspection.csv",
            "text/csv",
        )
        d2.download_button(
            "📥 Download OpenCV Image (PNG)",
            png_bytes(views["Annotated Result"]),
            "opencv_annotated.png",
            "image/png",
        )
    else:
        st.warning("No regions detected. Try toggling Smart Auto-Tuning in the sidebar.")

# ---------------------------------------------------------
# Tab 2: YOLO
# ---------------------------------------------------------
with tab_yolo:
    st.markdown("### 🤖 YOLOv8 AI Object Detection")
    if YOLO is None:
        st.error("Ultralytics package is missing. Run `pip install ultralytics`.")
    else:
        try:
            yolo_img, yolo_dets, yolo_latency = yolo_analysis(
                original, model_name, yolo_threshold, iou_threshold, img_size
            )

            yh_count = sum(d.review_priority == "HIGH" for d in yolo_dets)
            ym_count = sum(d.review_priority == "MEDIUM" for d in yolo_dets)

            if yh_count > 0:
                st.error(
                    f"🚨 **【HIGH RISK HAZARD ALERT】 YOLO detected {yh_count} sharp hazard(s) (Scissors/Knives/Syringes)! Highlighted in RED. Halt conveyor belt immediately!**"
                )

            yc1, yc2 = st.columns(2)
            yc1.image(
                to_rgb(original),
                caption=f"Input Image: {uploaded.name}",
                use_container_width=True,
            )
            yc2.image(
                yolo_img,
                caption=f"YOLOv8 ({model_name}) Output",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            ym1, ym2, ym3, ym4 = st.columns(4)
            with ym1:
                render_metric_card("Total AI Objects", len(yolo_dets), "total")
            with ym2:
                render_metric_card("High Hazard Sharps", yh_count, "high")
            with ym3:
                render_metric_card("Common Recyclables", ym_count, "medium")
            with ym4:
                render_metric_card("Inference Latency", f"{yolo_latency:.1f} ms", "speed")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 YOLO Inspection Table")
            if yolo_dets:
                df_yolo = pd.DataFrame([asdict(d) for d in yolo_dets])
                st.dataframe(df_yolo, use_container_width=True, hide_index=True)

                yd1, yd2 = st.columns(2)
                yd1.download_button(
                    "📥 Download YOLO CSV Report",
                    df_yolo.to_csv(index=False).encode("utf-8"),
                    "yolo_inspection.csv",
                    "text/csv",
                )
                yd2.download_button(
                    "📥 Download YOLO Image (PNG)",
                    png_bytes(yolo_img),
                    "yolo_annotated.png",
                    "image/png",
                )
            else:
                st.warning("No objects detected. Lower 'Confidence Threshold' in the sidebar.")
        except Exception as err:
            st.error(f"YOLO Execution Error: {err}")

# ---------------------------------------------------------
# Tab 3: Benchmark
# ---------------------------------------------------------
with tab_benchmark:
    st.markdown("### ⚡ Side-by-Side Algorithm Benchmark")
    bm_c1, bm_c2 = st.columns(2)
    bm_c1.image(
        views["Annotated Result"],
        caption=f"Classical OpenCV ({cv_latency:.1f} ms)",
        use_container_width=True,
    )
    if YOLO is not None:
        bm_c2.image(
            yolo_img,
            caption=f"YOLOv8 AI ({yolo_latency:.1f} ms)",
            use_container_width=True,
        )

    st.markdown("#### 📊 Metric Summary Comparison")
    comp_df = pd.DataFrame([
        {
            "Mode": "Classical OpenCV",
            "Objects Found": len(classical_dets),
            "High Hazards": sum(
                d.review_priority == "HIGH" for d in classical_dets
            ),
            "Latency (ms)": f"{cv_latency:.2f}",
            "Requires GPU": "No",
            "Zero-Training": "Yes",
        },
        {
            "Mode": f"YOLOv8 ({model_name})",
            "Objects Found": len(yolo_dets) if YOLO else 0,
            "High Hazards": sum(
                d.review_priority == "HIGH" for d in yolo_dets
            )
            if YOLO
            else 0,
            "Latency (ms)": f"{yolo_latency:.2f}" if YOLO else "N/A",
            "Requires GPU": "Recommended",
            "Zero-Training": "No (Pretrained/Fine-tuned)",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

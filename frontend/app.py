"""
Vehicle Smoke Detection System — frontend.

This file is UI ONLY: page layout, sidebar controls, image/video upload
handling, and rendering results. All detection/classification logic lives
in the backend/ package and is imported below — this file never touches
YOLO, Florence-2, or OpenCV model logic directly.
"""

import os
import sys
import tempfile
import time
import copy
import base64
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make the sibling `backend/` package importable regardless of the working
# directory `streamlit run` was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / "support" / "runtime" / "ultralytics").resolve()))


def format_color_percentages(color_percentages):
    if not color_percentages:
        return "Not available"

    labels = {
        "black_smoke": "Black",
        "white_smoke": "White",
        "grey_smoke": "Grey",
        "blue_smoke": "Blue",
    }
    return " | ".join(
        f"{label}: {color_percentages.get(class_name, 0.0):.1f}%"
        for class_name, label in labels.items()
    )


def show_video_progress_bar(progress_slot, progress_value):
    progress_percent = max(0.0, min(float(progress_value) * 100.0, 100.0))
    progress_slot.markdown(
        f"""
        <div class="video-progress-track">
            <div class="video-progress-fill" style="width: {progress_percent:.2f}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_data_uri(image_path):
    if not image_path or not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONE_NETWORK_LOGO_URI = image_to_data_uri(PROJECT_ROOT / "frontend" / "assets" / "one-network-logo.png")


# -----------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------
st.set_page_config(
    page_title="Vehicle Smoke Detection System",
    page_icon="ON",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------
# CUSTOM CSS
# -----------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; color: #111827; }
    .block-container {
        max-width: 1580px;
        padding-top: 0.7rem;
        padding-bottom: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    .stApp {
        background:
            linear-gradient(180deg, #f8fafc 0%, #f3f6fa 48%, #f7f9fc 100%) !important;
    }
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }
    header { background: transparent !important; box-shadow: none !important; }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: #111827; letter-spacing: 0; }
    hr { display: none; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 16px;
        margin-bottom: 12px;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #4b5563; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * { color: #111827; }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] span,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] span { color: #111827 !important; }
    [data-testid="stRadio"] {
        background: #ffffff;
        border: 1px solid #c7d0dd;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.07);
    }
    [data-testid="stSlider"] {
        background: #ffffff;
        border: 1px solid #c7d0dd;
        border-radius: 8px;
        padding: 14px 16px 20px 16px;
        margin-bottom: 16px;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.07);
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #c5cedb;
        border-radius: 8px;
        padding: 16px;
        min-height: 144px;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
        margin: 0 0 18px 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    [data-testid="stFileUploader"] > label {
        display: none !important;
    }
    [data-testid="stFileUploader"] section {
        background: #f8fafc;
        border: 1px dashed #aeb9c8;
        border-radius: 8px;
        min-height: 86px;
        padding: 14px 18px;
        transition: border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
    }
    [data-testid="stFileUploader"] section:hover {
        background: #f9fafb;
        border-color: #ef4444;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.08);
    }
    [data-testid="stFileUploader"] section > div {
        min-height: 54px !important;
        gap: 16px !important;
        align-items: center !important;
        flex-wrap: wrap !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 78px !important;
        padding: 10px 12px !important;
        align-items: center !important;
    }
    [data-testid="stFileUploaderDropzone"] > div:first-child {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #374151 !important;
        font-size: 14px !important;
        line-height: 1.3 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        display: none !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div:first-child,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div:first-child *,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileIcon"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileIcon"] * {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        color: #111827 !important;
        fill: #111827 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        background: #ffffff !important; color: #111827 !important;
        border-color: #d1d5db !important; font-weight: 800 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button {
        min-width: 108px !important;
        min-height: 40px !important;
        width: auto !important;
        height: 40px !important;
        padding: 0 16px !important;
        box-shadow: 0 4px 12px rgba(17, 24, 39, 0.08) !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg *,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [role="img"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [role="img"] * {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        opacity: 0 !important;
        color: transparent !important;
        fill: transparent !important;
        stroke: transparent !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
    [data-testid="stFileUploader"] button[aria-label*="Remove"],
    [data-testid="stFileUploader"] button[title*="Remove"] {
        min-width: 30px !important;
        min-height: 30px !important;
        width: 30px !important;
        height: 30px !important;
        padding: 0 !important;
        background: #f3f6fa !important;
        border: 1px solid #d8dee8 !important;
        border-radius: 999px !important;
        box-shadow: none !important;
        position: relative !important;
        overflow: hidden !important;
        color: transparent !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button svg,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button svg path,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button *,
    [data-testid="stFileUploader"] button[aria-label*="Remove"] *,
    [data-testid="stFileUploader"] button[title*="Remove"] * {
        display: none !important;
        color: transparent !important;
        fill: transparent !important;
        stroke: transparent !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button::before,
    [data-testid="stFileUploader"] button[aria-label*="Remove"]::before,
    [data-testid="stFileUploader"] button[title*="Remove"]::before {
        content: "×";
        position: absolute;
        inset: 0;
        display: flex !important;
        align-items: center;
        justify-content: center;
        color: #6b7280 !important;
        font-size: 18px;
        font-weight: 850;
        line-height: 1;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover {
        background: #fee2e2 !important;
        border-color: #fca5a5 !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover::before {
        color: #b91c1c !important;
    }
    [data-testid="stFileUploaderDropzone"] button svg,
    [data-testid="stFileUploaderDropzone"] button svg path {
        color: #111827 !important;
        fill: #111827 !important;
        stroke: #111827 !important;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:hover * {
        background: #f9fafb !important; color: #111827 !important;
        border-color: #ef4444 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important; border: 1px solid #9ca3af !important; border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: #111827 !important; fill: #111827 !important;
    }
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] { background: #ffffff !important; border: 1px solid #d1d5db !important; }
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] li div,
    [role="option"],
    [role="option"] * { color: #111827 !important; background: #ffffff !important; }
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover,
    [role="option"][aria-selected="true"] { color: #111827 !important; background: #e5e7eb !important; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(17, 24, 39, 0.05);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        background: #1f2937; color: #ffffff !important; border: 1px solid #1f2937;
        border-radius: 6px; font-weight: 700;
    }
    .stButton > button *,
    [data-testid="stDownloadButton"] > button * { color: #ffffff !important; }
    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        background: #374151; color: #ffffff !important; border-color: #374151;
    }
    .stButton > button,
    .stButton > button *,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stDownloadButton"] > button * { color: #ffffff !important; }
    [data-testid="stText"], [data-testid="stText"] *, pre, code { color: #111827 !important; opacity: 1 !important; }
    [data-testid="stProgress"] {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 12px 0 18px 0 !important;
    }
    [data-testid="stProgress"] div {
        color: #111827 !important;
    }
    [data-testid="stProgress"] > div {
        height: 7px !important;
        background: #d8dee8 !important;
        border-radius: 999px !important;
        overflow: hidden !important;
        box-shadow: inset 0 0 0 1px rgba(17, 24, 39, 0.08) !important;
        padding: 0 !important;
    }
    [data-testid="stProgress"] > div > div {
        height: 7px !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stProgress"] [role="progressbar"] {
        height: 7px !important;
        background: linear-gradient(90deg, #ef4444, #b91c1c) !important;
        border-radius: 999px !important;
        box-shadow: none !important;
    }
    .video-progress-track {
        width: 100%;
        height: 7px;
        background: #d8dee8;
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 999px;
        overflow: hidden;
        margin: 12px 0 18px 0;
        box-shadow: inset 0 1px 2px rgba(17, 24, 39, 0.08);
    }
    .video-progress-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #ef4444, #b91c1c);
        border-radius: 999px;
        transition: width 180ms ease;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] p { color: #111827 !important; opacity: 1 !important; }
    .app-header {
        background: #ffffff;
        border: 1px solid #c7d0dd;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        box-shadow: 0 10px 24px rgba(17, 24, 39, 0.07);
        position: relative;
        z-index: 5000;
        overflow: visible;
    }
    .app-title-group {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }
    .one-network-logo {
        width: 72px;
        height: auto;
        display: block;
        flex: 0 0 auto;
    }
    .title-text {
        text-align: left; font-size: 19px; font-weight: 850; color: #111827; margin: 0;
    }
    .subtitle-text {
        text-align: left; font-size: 10px; color: #6b7280; margin: 1px 0 0 0;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .app-header-status {
        border: 1px solid #c7d0dd;
        background: #f8fafc;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 800;
        color: #374151;
        white-space: nowrap;
        position: relative;
        cursor: help;
        overflow: visible;
        transition: transform 170ms ease, box-shadow 170ms ease, border-color 170ms ease, background 170ms ease;
    }
    .app-header-status:hover {
        background: #ffffff;
        border-color: rgba(239, 68, 68, 0.38);
        transform: translateY(-2px);
        box-shadow:
            0 14px 28px rgba(239, 68, 68, 0.16),
            0 4px 10px rgba(17, 24, 39, 0.06);
    }
    .app-header-status::after {
        content: attr(data-tooltip);
        position: absolute;
        right: 0;
        top: calc(100% + 10px);
        width: 340px;
        max-width: 80vw;
        background: #111827;
        color: #ffffff;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 12px;
        font-weight: 600;
        line-height: 1.45;
        white-space: normal;
        opacity: 0;
        transform: translateY(-4px);
        pointer-events: none;
        transition: opacity 160ms ease, transform 160ms ease;
        box-shadow: 0 16px 34px rgba(17, 24, 39, 0.22);
        z-index: 2147483647;
    }
    .app-header-status::before {
        content: "";
        position: absolute;
        right: 22px;
        top: calc(100% + 3px);
        width: 14px;
        height: 14px;
        background: #111827;
        transform: rotate(45deg);
        opacity: 0;
        pointer-events: none;
        transition: opacity 160ms ease;
        z-index: 2147483647;
    }
    .app-header-status:hover::after {
        opacity: 1;
        transform: translateY(0);
    }
    .app-header-status:hover::before {
        opacity: 1;
    }
    .sidebar-status-card {
        background: #f8fafc;
        border: 1px solid #c7d0dd;
        border-radius: 8px;
        padding: 12px;
        margin-top: 12px;
        box-shadow: 0 8px 20px rgba(17, 24, 39, 0.07);
    }
    .settings-panel-title {
        font-size: 18px;
        font-weight: 850;
        color: #111827;
        margin: 0 0 12px 0;
    }
    .sidebar-status-title {
        font-size: 12px;
        font-weight: 850;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .sidebar-status-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 6px 0;
        border-bottom: 1px solid #e5e7eb;
        font-size: 13px;
    }
    .sidebar-status-row:last-child { border-bottom: none; }
    .sidebar-status-row span:first-child { color: #6b7280; }
    .sidebar-status-row span:last-child { color: #111827; font-weight: 800; text-align: right; }
    .positive-banner {
        background: #991b1b; border-radius: 8px; padding: 18px; text-align: center;
        color: white; font-size: 20px; font-weight: 800; margin: 10px 0;
    }
    .negative-banner {
        background: #166534; border-radius: 8px; padding: 18px; text-align: center;
        color: white; font-size: 20px; font-weight: 800; margin: 10px 0;
    }
    .review-banner {
        background: #92400e; border-radius: 8px; padding: 18px; text-align: center;
        color: white; font-size: 20px; font-weight: 800; margin: 10px 0;
    }
    .smoke-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 18px;
        margin: 12px 0;
        border: 1px solid #d8dee8;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.05);
    }
    .info-row {
        display: flex; justify-content: space-between; gap: 16px; padding: 7px 0;
        border-bottom: 1px solid #e5e7eb; font-size: 14px;
    }
    .section-header {
        font-size: 15px; font-weight: 850; color: #111827;
        text-transform: uppercase; letter-spacing: 1px; margin: 22px 0 10px 0;
    }
    .upload-panel-title {
        font-size: 18px;
        font-weight: 850;
        color: #111827;
        margin: 0 0 12px 0;
    }
    .media-label {
        font-size: 15px;
        font-weight: 800;
        margin: 6px 0 8px 0;
        color: #111827;
    }
    [data-testid="stImage"],
    [data-testid="stVideo"] {
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stImage"] img,
    [data-testid="stVideo"] video {
        border-radius: 8px;
        border: 1px solid #d8dee8;
        box-shadow: 0 14px 32px rgba(17, 24, 39, 0.08);
        background: #ffffff;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.05);
        background: #ffffff;
        margin-bottom: 16px;
    }
    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid #d8dee8;
    }
    .stream-note {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
        padding: 12px 14px; color: #1e3a8a; font-weight: 600; margin: 8px 0 12px 0;
    }
    .status-card {
        background: #111827; color: #ffffff !important; border: 1px solid #111827;
        border-radius: 8px; padding: 12px 14px; font-size: 15px; font-weight: 800; margin: 12px 0;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.12);
    }
    .status-card * { color: #ffffff !important; }
    .display-status {
        background: #111827; color: #ffffff !important; border-radius: 8px; padding: 7px 10px;
        font-size: 12px; font-weight: 800; display: block; width: fit-content;
        margin: 8px 0 0 auto; max-width: calc(100% - 28px);
        border-left: 5px solid #22c55e; box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
    }
    .display-status.alert { border-left-color: #ef4444; }
    .display-status * { color: #ffffff !important; }
    .placeholder-box {
        background: #f9fafb; border: 1px dashed #9ca3af; border-radius: 8px; padding: 28px 16px;
        text-align: center; color: #374151 !important; font-weight: 700; min-height: 160px;
        display: flex; align-items: center; justify-content: center;
    }
    .placeholder-box * { color: #374151 !important; }
    .positive-banner, .negative-banner, .review-banner,
    .positive-banner *, .negative-banner *, .review-banner * { color: #ffffff !important; }
    .video-event-card {
        display: grid;
        grid-template-columns: minmax(0, 4fr) minmax(220px, 1fr);
        gap: 18px;
        align-items: stretch;
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 18px;
        margin: 14px 0;
        box-shadow: 0 14px 34px rgba(17, 24, 39, 0.06);
    }
    .video-event-details { min-width: 0; }
    .video-frame-box {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        background: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 230px;
    }
    .video-frame-box img {
        width: 100%;
        border-radius: 6px;
        display: block;
    }
    .video-frame-box a {
        cursor: zoom-in;
        display: block;
        width: 100%;
    }
    .image-lightbox-toggle {
        display: none;
    }
    .image-lightbox-thumb {
        cursor: zoom-in;
        display: block;
        width: 100%;
    }
    .image-lightbox-overlay {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 32px;
        background: rgba(17, 24, 39, 0.72);
        backdrop-filter: blur(5px);
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        transition: opacity 180ms ease;
    }
    .image-lightbox-overlay img {
        max-width: min(92vw, 1200px);
        max-height: 88vh;
        width: auto;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
        transform: scale(0.96);
        transition: transform 180ms ease;
    }
    .image-lightbox-toggle:checked + .image-lightbox-thumb + .image-lightbox-overlay {
        opacity: 1;
        visibility: visible;
        pointer-events: auto;
    }
    .image-lightbox-toggle:checked + .image-lightbox-thumb + .image-lightbox-overlay img {
        transform: scale(1);
    }
    .video-frame-placeholder {
        color: #6b7280;
        font-size: 14px;
        text-align: center;
    }
    .model-loading-panel {
        min-height: min(560px, calc(100vh - 190px));
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 18px 0 24px 0;
        overflow: hidden;
        position: relative;
        box-shadow: 0 18px 44px rgba(17, 24, 39, 0.07);
    }
    .model-loading-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, rgba(17, 24, 39, 0.04) 1px, transparent 1px),
            linear-gradient(0deg, rgba(17, 24, 39, 0.04) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(to bottom, rgba(0,0,0,0.75), transparent);
    }
    .model-loading-content {
        width: min(560px, 92%);
        position: relative;
        z-index: 1;
        text-align: center;
    }
    .loading-stage {
        height: 126px;
        position: relative;
        margin: 0 auto 18px auto;
    }
    .loading-road {
        position: absolute;
        left: 50%;
        bottom: 22px;
        width: 360px;
        max-width: 88%;
        height: 4px;
        transform: translateX(-50%);
        background: #d1d5db;
        border-radius: 999px;
        overflow: hidden;
    }
    .loading-road::after {
        content: "";
        position: absolute;
        inset: 0;
        width: 45%;
        background: #ef4444;
        border-radius: 999px;
        animation: road-scan 1.6s ease-in-out infinite;
    }
    .loading-car {
        position: absolute;
        left: 50%;
        bottom: 34px;
        width: 126px;
        height: 52px;
        transform: translateX(-50%);
        animation: car-idle 1.8s ease-in-out infinite;
    }
    .car-body {
        position: absolute;
        left: 9px;
        bottom: 9px;
        width: 108px;
        height: 30px;
        background: #111827;
        border-radius: 8px 14px 8px 8px;
    }
    .car-roof {
        position: absolute;
        left: 35px;
        bottom: 32px;
        width: 48px;
        height: 22px;
        background: #374151;
        border-radius: 14px 14px 4px 4px;
    }
    .car-wheel {
        position: absolute;
        bottom: 2px;
        width: 18px;
        height: 18px;
        background: #ffffff;
        border: 5px solid #111827;
        border-radius: 50%;
    }
    .car-wheel.left { left: 25px; }
    .car-wheel.right { right: 24px; }
    .smoke-trail {
        position: absolute;
        right: 11px;
        bottom: 34px;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: rgba(107, 114, 128, 0.34);
        animation: smoke-drift 1.7s ease-out infinite;
    }
    .smoke-trail.two { animation-delay: 0.35s; }
    .smoke-trail.three { animation-delay: 0.7s; }
    .model-loading-title {
        font-size: 22px;
        font-weight: 850;
        color: #111827;
        margin: 0 0 6px 0;
    }
    .model-loading-subtitle {
        font-size: 14px;
        color: #4b5563;
        margin: 0;
    }
    .loading-dots {
        display: flex;
        justify-content: center;
        gap: 7px;
        margin-top: 18px;
    }
    .loading-dots span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #ef4444;
        animation: loading-dot 1.2s ease-in-out infinite;
    }
    .loading-dots span:nth-child(2) { animation-delay: 0.18s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.36s; }
    @keyframes road-scan {
        0% { transform: translateX(-110%); }
        50% { transform: translateX(70%); }
        100% { transform: translateX(230%); }
    }
    @keyframes car-idle {
        0%, 100% { transform: translateX(-50%) translateY(0); }
        50% { transform: translateX(-50%) translateY(-3px); }
    }
    @keyframes smoke-drift {
        0% { opacity: 0; transform: translate(0, 0) scale(0.6); }
        25% { opacity: 0.75; }
        100% { opacity: 0; transform: translate(48px, -28px) scale(2.4); }
    }
    @keyframes loading-dot {
        0%, 100% { opacity: 0.35; transform: translateY(0); }
        50% { opacity: 1; transform: translateY(-5px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------
# LOAD MODELS (Streamlit caching lives here, not in backend)
# -----------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector_model():
    try:
        return backend.load_yolo_model()
    except backend.ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()


@st.cache_resource(show_spinner=False)
def get_classifier_model():
    device_info = backend.detect_device()
    return backend.load_florence_model(device_info["inference_device"])


@st.cache_resource(show_spinner=False)
def get_label_font():
    return backend.load_label_font()


loading_screen = st.empty()
try:
    with loading_screen.container():
        st.markdown(
            """
            <div class="model-loading-panel">
                <div class="model-loading-content">
                    <div class="loading-stage" aria-label="Loading smoke classifier">
                        <div class="smoke-trail"></div>
                        <div class="smoke-trail two"></div>
                        <div class="smoke-trail three"></div>
                        <div class="loading-car">
                            <div class="car-roof"></div>
                            <div class="car-body"></div>
                            <div class="car-wheel left"></div>
                            <div class="car-wheel right"></div>
                        </div>
                        <div class="loading-road"></div>
                    </div>
                    <div class="model-loading-title">Preparing Smoke Analysis</div>
                    <p class="model-loading-subtitle">Loading detection and classification models...</p>
                    <div class="loading-dots"><span></span><span></span><span></span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    import backend.app as backend

    device_info = backend.detect_device()
    detector_model, detector_model_name = get_detector_model()
    classifier_model, classifier_processor, classifier_source = get_classifier_model()
    label_font = get_label_font()
    loading_screen.empty()
except OSError as exc:
    loading_screen.empty()
    st.error("Florence-2 could not load. Classification is required for this app, so processing cannot continue.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------
# TITLE
# -----------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title-group">
            <img class="one-network-logo" src="{ONE_NETWORK_LOGO_URI}" alt="One Network logo">
            <div>
                <div class="title-text">Vehicle Smoke Detection System</div>
                <div class="subtitle-text">AI-powered exhaust smoke detection</div>
            </div>
        </div>
        <div class="app-header-status" data-tooltip="This module detects visible vehicle exhaust smoke, estimates smoke type, and prepares inspection reports. It helps vehicle fitness teams find high-emission vehicles faster, improving air quality and supporting climate-change reduction efforts.">Smoke Analysis Module</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------
# PERMANENT SETTINGS PANEL
# -----------------------------------------------
settings_panel, workspace_panel = st.columns([0.20, 0.80], gap="medium")

with settings_panel:
    st.markdown('<div class="settings-panel-title">Detection Settings</div>', unsafe_allow_html=True)

    detector_conf = st.slider(
        "Smoke detector confidence",
        min_value=0.10, max_value=1.00, value=0.50, step=0.05,
        key="smoke_detector_confidence_50_default",
        help="Higher means fewer but stronger smoke detections.",
    )
    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10, max_value=0.90, value=0.60, step=0.05,
        help="Controls duplicate/overlapping boxes. Higher = less aggressive merging.",
    )
    video_analysis_fps = 10

    mode = st.radio("Input Mode", ["Image", "Video"])
    st.markdown(
        f"""
        <div class="sidebar-status-card">
            <div class="sidebar-status-title">System Status</div>
            <div class="sidebar-status-row"><span>Module</span><span>Smoke Detection</span></div>
            <div class="sidebar-status-row"><span>Detector</span><span>{detector_model_name}</span></div>
            <div class="sidebar-status-row"><span>Classifier</span><span>Florence-2</span></div>
            <div class="sidebar-status-row"><span>Device</span><span>{device_info['device_label']}</span></div>
            <div class="sidebar-status-row"><span>CUDA</span><span>{device_info['cuda_version']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Every backend call needs this — built once per script run from the
# sidebar's current values, then handed to the engine.

config = backend.DetectionConfig(
    detector_conf=detector_conf,
    iou_threshold=iou_threshold,
    video_analysis_fps=video_analysis_fps,
)
report_interval_seconds = config.report_interval_seconds

engine = backend.SmokeDetectionEngine(
    detector_model=detector_model,
    classifier_model=classifier_model,
    classifier_processor=classifier_processor,
    inference_device=device_info["inference_device"],
    use_half_precision=device_info["use_half_precision"],
    config=config,
)


# -----------------------------------------------
# REPORT RENDERING (UI only — pulls data from backend.get_smoke_info)
# -----------------------------------------------
def show_vehicle_result(detections, timing_rows=None, total_time_seconds=None):
    if not detections:
        st.markdown('<div class="negative-banner">NO SMOKE DETECTED</div>', unsafe_allow_html=True)
        return

    statuses = [backend.get_smoke_info(d["class_name"])["status"] for d in detections]
    if "UNFIT" in statuses:
        st.markdown('<div class="positive-banner">SMOKE DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="review-banner">SMOKE DETECTED</div>', unsafe_allow_html=True)

    avg_detector = np.mean([d["detector_conf"] for d in detections]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Smoke Areas", len(detections))
    c2.metric("Average Smoke Confidence", f"{avg_detector:.1f}%")
    if total_time_seconds is None:
        c3.metric("Alert Time", datetime.now().strftime("%H:%M:%S"))
    else:
        c3.metric("Total Process Time", f"{total_time_seconds:.2f}s")

    if timing_rows:
        st.markdown('<div class="section-header">Processing Time</div>', unsafe_allow_html=True)
        st.dataframe(timing_rows, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Smoke Alert</div>', unsafe_allow_html=True)
    st.dataframe(
        [{
            "Alert": "Smoke detected",
            "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Smoke Boxes": len(detections),
            "Average Confidence": f"{avg_detector:.1f}%",
            "Message": "Smoke emission detected in the uploaded image.",
        }],
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="section-header">Detection Report</div>', unsafe_allow_html=True)
    for index, detection in enumerate(detections, start=1):
        info = backend.get_smoke_info(detection["class_name"])
        color_percentages = format_color_percentages(detection.get("color_percentages"))
        st.markdown(
            f"""
            <div class="smoke-card" style="border-color:{info['color']};">
                <div style="font-size:18px; font-weight:800; color:{info['color']}; margin-bottom:10px;">
                    Detection {index} - {info['label']}
                </div>
                <div class="info-row"><span>Smoke Type</span><span><b>{info['label']}</b></span></div>
                <div class="info-row"><span>Detection Confidence</span><span><b>{detection['detector_conf'] * 100:.1f}%</b></span></div>
                <div class="info-row"><span>Color Percentages</span><span><b>{color_percentages}</b></span></div>
                <div class="info-row"><span>Severity</span><span><b>{info['severity']}</b></span></div>
                <div class="info-row"><span>Probable Cause</span><span>{info['cause']}</span></div>
                <div class="info-row" style="border-bottom:none"><span>Recommended Action</span><span>{info['action']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def show_live_video_report(smoke_events):
    if not smoke_events:
        return

    st.markdown('<div class="section-header">Vehicle Smoke Event Report</div>', unsafe_allow_html=True)

    for event_index, event in enumerate(smoke_events, start=1):
        smoke_type = event.get("Smoke Type", "Smoke") or "Smoke"
        first_smoke_type = smoke_type.split(",")[0].strip()
        info = backend.get_smoke_info(first_smoke_type)
        frame_image_path = event.get("Frame Image", "")
        frame_image_uri = image_to_data_uri(frame_image_path)
        lightbox_id = f"video-report-frame-{event_index}"
        frame_html = (
            f'<input class="image-lightbox-toggle" id="{lightbox_id}" type="checkbox">'
            f'<label class="image-lightbox-thumb" for="{lightbox_id}" title="Click to enlarge">'
            f'<img src="{frame_image_uri}" alt="Saved YOLO frame"></label>'
            f'<label class="image-lightbox-overlay" for="{lightbox_id}">'
            f'<img src="{frame_image_uri}" alt="Enlarged saved YOLO frame"></label>'
            if frame_image_uri
            else '<div class="video-frame-placeholder">Saved frame unavailable</div>'
        )
        st.markdown(
            f"""
            <div class="video-event-card">
                <div class="video-event-details">
                    <div style="font-size:18px; font-weight:800; color:{info['color']}; margin-bottom:10px;">
                        Detection {event_index} - {smoke_type}
                    </div>
                    <div class="info-row"><span>Smoke Type</span><span><b>{smoke_type}</b></span></div>
                    <div class="info-row"><span>Detection Confidence</span><span><b>{event.get('Smoke Detection %', '')}</b></span></div>
                    <div class="info-row"><span>Color Percentages</span><span><b>{event.get('Color Percentages', 'Not available')}</b></span></div>
                    <div class="info-row"><span>Severity</span><span><b>{info['severity']}</b></span></div>
                    <div class="info-row"><span>Probable Cause</span><span>{info['cause']}</span></div>
                    <div class="info-row"><span>Recommended Action</span><span>{info['action']}</span></div>
                    <div class="info-row"><span>Video Time</span><span>{event.get('Video Time', '')}</span></div>
                    <div class="info-row" style="border-bottom:none"><span>Frame</span><span>{event.get('Frame', '')}</span></div>
                </div>
                <div class="video-frame-box">
                    {frame_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_video_detection_frame_table(detection_rows):
    if not detection_rows:
        return

    table_columns = [
        "Alert",
        "Date",
        "First Seen Time",
        "Real Date/Time",
        "Video Time",
        "Video Seconds",
        "Frame",
        "Smoke Boxes",
        "Smoke Detection %",
        "Smoke Type",
        "Report Type",
    ]
    visible_rows = [
        {column: row.get(column, "") for column in table_columns}
        for row in detection_rows
    ]

    st.markdown('<div class="section-header">Smoke Detection Frame Table</div>', unsafe_allow_html=True)
    st.dataframe(visible_rows, use_container_width=True, hide_index=True)


with workspace_panel:
    # -----------------------------------------------
    # IMAGE MODE
    # -----------------------------------------------
    if mode == "Image":
        st.markdown('<div class="upload-panel-title">Image Analysis</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a vehicle image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            total_process_start = time.perf_counter()
            image = Image.open(uploaded_file).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting smoke..."):
                detection_start = time.perf_counter()
                detections = engine.detect_smoke_boxes(image_bgr)
                detection_time_seconds = time.perf_counter() - detection_start
                annotated_bgr = backend.draw_detections(image_bgr, detections, label_font)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="media-label">Original Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col2:
                st.markdown('<div class="media-label">Detection Output</div>', unsafe_allow_html=True)
                image_output_slot = st.empty()
                image_output_slot.image(annotated_rgb, use_container_width=True)

            report_slot = st.empty()
            timing_rows = [
                {
                    "Step": "YOLO smoke detection",
                    "Smoke": "All",
                    "Time": f"{detection_time_seconds:.2f}s",
                }
            ]

            with report_slot.container():
                if detections:
                    avg_detector_conf = np.mean([d["detector_conf"] for d in detections]) * 100
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Smoke Areas", len(detections))
                    p2.metric("Average Detection Confidence", f"{avg_detector_conf:.1f}%")
                    p3.metric("Detection Time", f"{detection_time_seconds:.2f}s")
                    st.info(f"Smoke detected in {len(detections)} area(s). Classifying smoke type one crop at a time...")
                    st.dataframe(timing_rows, use_container_width=True, hide_index=True)
                else:
                    st.markdown('<div class="negative-banner">NO SMOKE DETECTED</div>', unsafe_allow_html=True)

            if detections:
                classified_detections = []
                classification_progress = st.progress(0)
                classification_status = st.empty()

                for index, detection in enumerate(detections, start=1):
                    classification_status.info(f"Classifying smoke crop {index} of {len(detections)}...")
                    classification_start = time.perf_counter()
                    classified_detection = engine.classify_one_detected_smoke(detection)
                    classification_time_seconds = time.perf_counter() - classification_start
                    classified_detections.append(classified_detection)

                    timing_rows.append(
                        {
                            "Step": "Florence smoke classification",
                            "Smoke": index,
                            "Time": f"{classification_time_seconds:.2f}s",
                        }
                    )
                    classification_progress.progress(index / len(detections))

                    with report_slot.container():
                        show_vehicle_result(
                            classified_detections,
                            timing_rows=timing_rows,
                            total_time_seconds=time.perf_counter() - total_process_start,
                        )
                        remaining = len(detections) - index
                        if remaining:
                            st.info(f"Still classifying {remaining} smoke crop(s)...")

                detections = classified_detections
                classification_status.empty()
                classification_progress.empty()

            with report_slot.container():
                show_vehicle_result(
                    detections,
                    timing_rows=timing_rows if detections else None,
                    total_time_seconds=time.perf_counter() - total_process_start,
                )


    # -----------------------------------------------
    # VIDEO MODE
    # -----------------------------------------------
    if mode == "Video":
        st.markdown('<div class="upload-panel-title">Video Analysis</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload a vehicle video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )

        if uploaded_video is not None:
            upload_key = f"{uploaded_video.name}-{uploaded_video.size}"
            if st.session_state.get("current_video_upload_key") != upload_key:
                st.session_state["current_video_upload_key"] = upload_key
                st.session_state.pop("processed_video_path", None)

            uploaded_video_bytes = uploaded_video.getvalue()
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_video_bytes)
            temp_video.flush()
            temp_video.close()

            input_col, output_col = st.columns(2)
            with input_col:
                st.markdown('<div class="media-label">Input Video</div>', unsafe_allow_html=True)
                st.video(temp_video.name)
            with output_col:
                st.markdown('<div class="media-label">Display Output</div>', unsafe_allow_html=True)
                display_video_slot = st.empty()
                if st.session_state.get("processed_video_path") and os.path.exists(st.session_state["processed_video_path"]):
                    display_video_slot.empty()
                    display_video_slot.video(st.session_state["processed_video_path"], autoplay=True, muted=True)
                else:
                    display_video_slot.video(temp_video.name)
                display_status_slot = st.empty()
                display_status_slot.markdown(backend.render_display_status([]), unsafe_allow_html=True)

            should_process_video = st.session_state.get("processed_video_upload_key") != upload_key
            if should_process_video:
                display_status_slot.markdown(backend.render_display_status([]), unsafe_allow_html=True)
                display_video_slot.video(temp_video.name)

                cap = cv2.VideoCapture(temp_video.name)
                if not cap.isOpened():
                    st.error("Could not open the uploaded video.")
                    st.stop()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if fps is None or fps <= 0:
                    fps = 25.0
                if width <= 0 or height <= 0:
                    st.error("Could not read the uploaded video's resolution.")
                    cap.release()
                    st.stop()

                frame_stride = backend.get_video_frame_stride(fps, video_analysis_fps)
                status_update_stride = max(1, int(round(fps / 2)))

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_result.mp4").name
                writer = backend.create_video_writer(output_path, fps, width, height)

                progress_slot = st.empty()
                show_video_progress_bar(progress_slot, 0)
                status_text = st.empty()
                live_report_slot = st.empty()
                detection_table_slot = st.empty()
                frame_count = 0
                smoke_frames = 0
                smoke_detection_count = 0
                last_detections = []
                detection_checks = 0
                processing_start_time = time.perf_counter()
                smoke_inference_times = []
                smoke_classification_times = []
                first_smoke_video_second = None
                first_smoke_wall_second = None
                first_smoke_pipeline_delay = None
                smoke_events = []
                video_detection_rows = []
                recent_smoke_sources = []
                last_rendered_events_count = 0
                classification_futures = []
                classification_executor = ThreadPoolExecutor(max_workers=2)
                last_report_second = -999.0
                saved_report_frames = 0
                report_frame_folder = backend.create_video_report_frame_folder(uploaded_video.name)
                display_update_stride = max(1, int(round(fps / 5)))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    should_analyze = frame_count % frame_stride == 0
                    if should_analyze:
                        smoke_start = time.perf_counter()
                        last_detections = engine.detect_smoke_boxes(frame)
                        smoke_inference_ms = (time.perf_counter() - smoke_start) * 1000
                        smoke_inference_times.append(smoke_inference_ms)
                        analysis_second = frame_count / fps

                        if last_detections and first_smoke_video_second is None:
                            first_smoke_video_second = analysis_second
                            first_smoke_wall_second = time.perf_counter() - processing_start_time
                            first_smoke_pipeline_delay = first_smoke_wall_second - first_smoke_video_second
                        detection_checks += 1

                        should_add_report = (
                            bool(last_detections)
                            and analysis_second - last_report_second >= report_interval_seconds
                        )
                        if should_add_report:
                            report_detections = copy.deepcopy(last_detections)
                            report_frame = backend.draw_fast_video_detections(frame, report_detections)
                            report_frame_path = backend.save_video_report_frame(
                                report_frame,
                                report_frame_folder,
                                frame_count,
                                analysis_second,
                            )
                            current_vehicle_caption = engine.describe_frame_with_vlm(report_frame_path)
                            recent_source = backend.find_recent_smoke_source(
                                current_vehicle_caption,
                                recent_smoke_sources,
                                analysis_second,
                            )
                            table_detections = copy.deepcopy(report_detections)
                            report_type = "New vehicle/source"
                            if recent_source:
                                table_detections = backend.copy_previous_classification(
                                    table_detections,
                                    recent_source["detections"],
                                )
                                report_type = "Repeated vehicle/source"

                            detection_row = backend.build_smoke_event(
                                frame_count,
                                fps,
                                table_detections,
                                source_name=uploaded_video.name,
                                classification_time_seconds=None,
                                frame_image_path=report_frame_path,
                            )
                            detection_row["Frame Caption"] = current_vehicle_caption
                            detection_row["Report Type"] = report_type
                            video_detection_rows.append(detection_row)
                            with detection_table_slot.container():
                                show_video_detection_frame_table(video_detection_rows)

                            def background_classification(detections_copy, frame_num, video_fps, source, frame_path, vehicle_caption):
                                c_start = time.perf_counter()
                                classified = engine.classify_detected_smoke(detections_copy)
                                c_time = time.perf_counter() - c_start
                                event = backend.build_smoke_event(
                                    frame_num,
                                    video_fps,
                                    classified,
                                    source_name=source,
                                    classification_time_seconds=c_time,
                                    frame_image_path=frame_path,
                                )
                                event["Frame Caption"] = vehicle_caption
                                return event, c_time, classified, vehicle_caption

                            if recent_source:
                                # Same vehicle/source was already reported recently, so do not
                                # create another report card or send it for classification again.
                                pass
                            else:
                                saved_report_frames += 1
                                backend.remember_smoke_source(
                                    recent_smoke_sources,
                                    analysis_second,
                                    report_detections,
                                    current_vehicle_caption,
                                )
                                future = classification_executor.submit(
                                    background_classification,
                                    report_detections,
                                    frame_count,
                                    fps,
                                    uploaded_video.name,
                                    report_frame_path,
                                    current_vehicle_caption,
                                )
                                classification_futures.append(future)
                            last_report_second = analysis_second

                        display_status_slot.markdown(backend.render_display_status(last_detections), unsafe_allow_html=True)

                    completed_futures = [future for future in classification_futures if future.done()]
                    if completed_futures:
                        for future in completed_futures:
                            event, c_time, classified_detections, vehicle_caption = future.result()
                            smoke_events.append(event)
                            smoke_classification_times.append(c_time)
                            for row in video_detection_rows:
                                same_finished_frame = row.get("Frame") == event.get("Frame")
                                same_caption_source = backend.captions_look_like_same_vehicle(
                                    row.get("Frame Caption", ""),
                                    vehicle_caption,
                                )
                                if same_finished_frame or same_caption_source:
                                    row.update({
                                        "Smoke Type": event.get("Smoke Type", row.get("Smoke Type", "")),
                                        "Color Percentages": event.get("Color Percentages", row.get("Color Percentages", "")),
                                        "Classification Time": event.get("Classification Time", ""),
                                    })
                            with detection_table_slot.container():
                                show_video_detection_frame_table(video_detection_rows)
                            backend.remember_smoke_source(
                                recent_smoke_sources,
                                event.get("Video Seconds", 0.0),
                                classified_detections,
                                vehicle_caption,
                            )
                            classification_futures.remove(future)

                    if len(smoke_events) > last_rendered_events_count:
                        smoke_events.sort(key=lambda item: item.get("Video Seconds", 0.0))
                        with live_report_slot.container():
                            show_live_video_report(smoke_events)
                        last_rendered_events_count = len(smoke_events)

                    detections = last_detections
                    if detections:
                        annotated = backend.draw_fast_video_detections(frame, detections)
                    else:
                        annotated = frame

                    if annotated.shape[1] != width or annotated.shape[0] != height:
                        annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_LINEAR)

                    writer.write(annotated)

                    if frame_count % display_update_stride == 0:
                        live_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        display_video_slot.image(
                            live_rgb, channels="RGB", use_container_width=True,
                        )

                    if detections:
                        smoke_frames += 1
                        smoke_detection_count += len(detections)

                    frame_count += 1
                    current_second = frame_count / fps
                    if frame_count % status_update_stride == 0 or frame_count == 1:
                        show_video_progress_bar(
                            progress_slot,
                            min(frame_count / max(total_frames, 1), 1.0),
                        )
                        status_text.markdown(
                            f"""<div class="status-card">
                                Processing frame {frame_count} of {total_frames}
                                ({current_second:.1f}s) | AI checks: {detection_checks}
                            </div>""",
                            unsafe_allow_html=True,
                        )

                cap.release()
                writer.release()

                if classification_futures:
                    status_text.markdown(
                        """<div class="status-card">
                            Finalizing video smoke classification reports...
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    for future in as_completed(classification_futures):
                        event, c_time, classified_detections, vehicle_caption = future.result()
                        smoke_events.append(event)
                        smoke_classification_times.append(c_time)
                        for row in video_detection_rows:
                            same_finished_frame = row.get("Frame") == event.get("Frame")
                            same_caption_source = backend.captions_look_like_same_vehicle(
                                row.get("Frame Caption", ""),
                                vehicle_caption,
                            )
                            if same_finished_frame or same_caption_source:
                                row.update({
                                    "Smoke Type": event.get("Smoke Type", row.get("Smoke Type", "")),
                                    "Color Percentages": event.get("Color Percentages", row.get("Color Percentages", "")),
                                    "Classification Time": event.get("Classification Time", ""),
                                })
                        with detection_table_slot.container():
                            show_video_detection_frame_table(video_detection_rows)
                        backend.remember_smoke_source(
                            recent_smoke_sources,
                            event.get("Video Seconds", 0.0),
                            classified_detections,
                            vehicle_caption,
                        )
                    classification_executor.shutdown()
                    smoke_events.sort(key=lambda item: item.get("Video Seconds", 0.0))
                    with live_report_slot.container():
                        show_live_video_report(smoke_events)

                status_text.empty()

                st.session_state["processed_video_path"] = output_path
                st.session_state["processed_video_upload_key"] = upload_key
                display_video_slot.empty()
                display_video_slot.video(output_path, autoplay=True, muted=True)

                if smoke_frames > 0:
                    st.markdown('<div class="positive-banner">SMOKE DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="negative-banner">NO SMOKE DETECTED</div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Frames", frame_count)
                c2.metric("Smoke Frames", smoke_frames)
                c3.metric("AI Checks", detection_checks)
                c4.metric("Saved Report Frames", saved_report_frames)

                if smoke_classification_times:
                    st.markdown('<div class="section-header">Smoke Classification Benchmark</div>', unsafe_allow_html=True)
                    st.dataframe(
                        [{
                            "Classification Reports": len(smoke_classification_times),
                            "Average Classification Time (s)": round(float(np.mean(smoke_classification_times)), 2),
                            "Fastest Classification Time (s)": round(float(np.min(smoke_classification_times)), 2),
                            "Slowest Classification Time (s)": round(float(np.max(smoke_classification_times)), 2),
                        }],
                        use_container_width=True, hide_index=True,
                    )

                # Keep these temp video files available for Streamlit's video player
                # during the current session.

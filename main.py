from dotenv import load_dotenv
load_dotenv()
import sqlite3
import cv2
import os
import datetime
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import time
import csv
import asyncio
import math
import matplotlib.pyplot as plt
import imageio
import psutil
import threading
import webbrowser



from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from utils.detector import Detector
from utils.tracker import SimpleTracker
from utils.incident_logic import IncidentDetector
from typing import List
from groq import Groq
from contextlib import asynccontextmanager
from asyncio import AbstractEventLoop
from reportlab.platypus import PageBreak
from reportlab.lib.units import mm
from collections import deque

print("THIS IS THE MAIN FILE BEING EXECUTED")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")
groq_client = Groq(api_key=api_key)

from sklearn.cluster import DBSCAN
from fastapi import FastAPI, Body, WebSocket
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
ZONES = {
    1: {"name": "Connaught Place", "lat": 28.6315, "lng": 77.2167, "traffic": "HIGH"},
    2: {"name": "India Gate", "lat": 28.6129, "lng": 77.2295, "traffic": "MEDIUM"},
    3: {"name": "Karol Bagh", "lat": 28.6519, "lng": 77.1909, "traffic": "HIGH"},
    4: {"name": "Lajpat Nagar", "lat": 28.5677, "lng": 77.2436, "traffic": "LOW"},
}
# ============================================
# FASTAPI SETUP
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(process_summary_batch())
    yield
    try:
        db_conn.close()
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DIRECTORIES
# ============================================

VIDEO_PATH = "input/big_video.mp4"
SAVE_PATH = "incidents"
os.makedirs(SAVE_PATH, exist_ok=True)

app.mount("/snapshots", StaticFiles(directory=SAVE_PATH), name="snapshots")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
# ============================================
# AI COMPONENTS
# ============================================

detector = Detector()
tracker = SimpleTracker()
incident_detector = IncidentDetector()

# ============================================
# DEVICE SETUP (CUDA / MPS / CPU)
# ============================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ============================================
# DEEP LEARNING MODEL (CNN + LSTM)
# ============================================

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        b, seq, C, H, W = x.size()
        cnn_out = []
        for t in range(seq):
            out = self.cnn(x[:, t])
            out = out.view(b, -1)
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out)
        final = lstm_out[:, -1, :]
        return self.fc(final)

model = CNN_LSTM().to(device)
model.load_state_dict(torch.load("incident_model.pth", map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# ============================================
# GLOBAL STATE
# ============================================

incident_logs = []

confidence_threshold = 0.7
heatmap_enabled = True
dl_enabled = True

collision_heatmap = None
stopped_heatmap = None
heatmap_decay = 0.995
main_loop: AbstractEventLoop | None = None
collision_persistence = {}

previous_centers = {}
frame_buffer = []
previous_frame_time = None
object_motion = {}
FPS_ASSUMED = 30
METERS_PER_PIXEL = 0.08   # adjust based on camera height
REPLAY_SECONDS = 2
REPLAY_FPS = 20
MAX_REPLAY_FRAMES = REPLAY_SECONDS * REPLAY_FPS
replay_buffer = deque(maxlen=MAX_REPLAY_FRAMES)
sequence_length = 16
alert_toggle = False
incident_density = {}
dispatch_log = {}
incident_cooldown = {}
COOLDOWN_SECONDS = 10
active_connections: List[WebSocket] = []
summary_queue = deque()
BATCH_SIZE = 2
BATCH_TIMEOUT = 8 #seconds
current_zone_id = 1
# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def analyze_root_cause(data, speed, overlap, stopped_time):
    causes = []
    if overlap > 0.5:
        causes.append("High vehicle overlap")
    if speed > 60:
        causes.append("Over-speeding")
    if stopped_time > 5:
        causes.append("Sudden braking")
    if not causes:
        causes.append("Manual review required")
    return causes


def risk_score_to_level(score):
    if score >= 160:
        return "CRITICAL"
    elif score >= 100:
        return "HIGH"
    elif score >= 60:
        return "MEDIUM"
    return "LOW"


def calculate_severity(confidence, vehicles):
    score = confidence * 50 + vehicles * 10
    if score > 70:
        return "HIGH"
    elif score > 40:
        return "MEDIUM"
    return "LOW"


def generate_ai_explanation(incident):
    """
    Smart AI narrative generator
    """

    causes = incident.get("root_causes", [])
    speed = incident.get("speed", 0)
    risk = incident.get("risk_level", "LOW")
    confidence = incident.get("confidence", 0)

    explanation_parts = []

    # Overlap-based reasoning
    if any("overlap" in c.lower() for c in causes):
        explanation_parts.append(
            "A sustained high vehicle overlap was detected, indicating probable physical contact."
        )

    # Speed-based reasoning
    if speed > 70:
        explanation_parts.append(
            "Vehicle was travelling at high speed, reducing reaction time."
        )
    elif speed > 40:
        explanation_parts.append(
            "Moderate speed may have contributed to delayed braking."
        )

    # Confidence reasoning
    if confidence > 0.9:
        explanation_parts.append(
            "Detection confidence is very high, indicating strong visual evidence."
        )

    # Risk reasoning
    if risk == "CRITICAL":
        explanation_parts.append(
            "Incident escalated to CRITICAL due to risk scoring, density clustering, or time-based escalation."
        )
    elif risk == "HIGH":
        explanation_parts.append(
            "Risk classified as HIGH based on combined motion, overlap, and contextual factors."
        )

    if not explanation_parts:
        explanation_parts.append(
            "Incident detected based on anomaly patterns in vehicle behavior."
        )

    final_summary = " ".join(explanation_parts)

    return final_summary


def generate_incident_report(incident):

    safe_time = str(incident["time"]).replace(":", "-").replace(" ", "_")
    filename = f"incident_report_{safe_time}.pdf"

    doc = SimpleDocTemplate(
        filename,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )
    elements = []

    styles = getSampleStyleSheet()
    custom_title_style = ParagraphStyle(
        name="CustomTitle",
        parent=styles["Title"],
        fontName="Helvetica",
        fontSize=22,
        textColor=colors.darkblue,
        spaceAfter=14
    )

    custom_normal_style = ParagraphStyle(
        name="CustomNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14
    )
    # ==========================================
    # HEADER
    # ==========================================

    elements.append(Paragraph("SMART CITY TRAFFIC CONTROL SYSTEM", custom_title_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("OFFICIAL INCIDENT REPORT", styles["Heading2"]))
    elements.append(Spacer(1, 20))

    # ==========================================
    # TIMELINE GRAPH SECTION
    # ==========================================

    elements.append(PageBreak())

    elements.append(Paragraph("Incident Timeline Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    timeline_path = generate_timeline_graph(incident)

    if timeline_path:
        img = Image(timeline_path, width=6 * inch, height=3 * inch)
        elements.append(img)
        elements.append(Spacer(1, 20))

    # ==========================================
    # DISPATCH TIMELINE
    # ==========================================

    elements.append(Paragraph("Dispatch Timeline Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    dispatch_path = generate_dispatch_timeline_graph(incident)

    if dispatch_path:
        img = Image(dispatch_path, width=6 * inch, height=2 * inch)
        elements.append(img)
        elements.append(Spacer(1, 20))

    # ==========================================
    # RISK BREAKDOWN GRAPH
    # ==========================================

    elements.append(Paragraph("Risk Score Breakdown Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    risk_path = generate_risk_breakdown_graph(incident)

    if risk_path:
        img = Image(risk_path, width=5 * inch, height=3 * inch)
        elements.append(img)
        elements.append(Spacer(1, 20))
    # ==========================================
    # BASIC INFO TABLE
    # ==========================================

    info_data = [
        ["Incident ID", str(incident["id"])],
        ["Date & Time", str(incident["time"])],
        ["Incident Type", incident["type"]],
        ["Risk Level", incident["risk_level"]],
        ["Risk Score", str(round(incident["risk_score"], 2))],
        ["Severity", incident["severity"]],
        ["Detection Confidence", str(round(incident["confidence"], 2))]
    ]

    table = Table(info_data, colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # ==========================================
    # INCIDENT SUMMARY
    # ==========================================

    elements.append(Paragraph("1. Incident Summary", styles["Heading3"]))
    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            incident.get("ai_explanation", "No AI summary available."),
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 20))

    # ==========================================
    # ROOT CAUSE ANALYSIS
    # ==========================================

    elements.append(Paragraph("2. Root Cause Analysis", styles["Heading3"]))
    elements.append(Spacer(1, 10))

    for cause in incident.get("root_causes", []):
        elements.append(Paragraph(f"• {cause}", styles["Normal"]))
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 20))

    # ==========================================
    # PREVENTION RECOMMENDATIONS
    # ==========================================

    elements.append(Paragraph("3. Prevention Recommendations", styles["Heading3"]))
    elements.append(Spacer(1, 10))

    for suggestion in incident.get("prevention_suggestions", []):
        elements.append(Paragraph(f"• {suggestion}", styles["Normal"]))
        elements.append(Spacer(1, 5))

    elements.append(Paragraph("4. Detailed Incident Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    detailed_summary = generate_detailed_summary(incident)
    elements.append(Spacer(1, 10))

    seen_lines = set()

    for line in detailed_summary.split("\n"):
        clean = line.strip()

        # Skip empty or repeated filler
        if not clean:
            continue
        if clean in seen_lines:
            continue

        seen_lines.add(clean)

        elements.append(Paragraph(clean, custom_normal_style))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("4. Comprehensive Forensic Investigation Report", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for line in incident.get("forensic_summary", "").split("\n"):
        elements.append(Paragraph(line.strip(), custom_normal_style))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 20))
    # ==========================================
    # SNAPSHOT IMAGE
    # ==========================================

    if incident.get("image"):
        try:
            image_path = incident["image"].replace("/snapshots/", "incidents/")
            img = Image(image_path, width=4 * inch, height=3 * inch)
            elements.append(Paragraph("4. Captured Snapshot", styles["Heading3"]))
            elements.append(Spacer(1, 10))
            elements.append(img)
            elements.append(Spacer(1, 20))
        except:
            pass

    # ==========================================
    # FOOTER
    # ==========================================

    elements.append(Paragraph("Report Generated By:", styles["Heading4"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("AI Surveillance Intelligence Module", styles["Normal"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("Smart City Traffic Monitoring Division", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("---- End of Report ----", styles["Italic"]))

    def add_page_number(canvas, doc):
        page_num_text = f"Page {doc.page}"
        canvas.drawRightString(190 * mm, 10 * mm, page_num_text)

    doc.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )
    return filename
def update_density(lat, lng):
    key = (round(lat, 3), round(lng, 3))
    incident_density[key] = incident_density.get(key, 0) + 1
    return incident_density[key]
def generate_prevention_tips(root_causes):
    tips = []

    for cause in root_causes:
        if "overlap" in cause.lower():
            tips.append("Maintain safe following distance between vehicles.")
        if "overspeed" in cause.lower():
            tips.append("Enforce speed monitoring and install speed cameras.")
        if "braking" in cause.lower():
            tips.append("Improve traffic signal timing and road visibility.")

    if not tips:
        tips.append("Manual traffic review recommended.")

    return tips

def generate_prevention_suggestions(incident):

    suggestions = []

    if incident["risk_level"] in ["HIGH", "CRITICAL"]:
        suggestions.append("Increase traffic monitoring at this location.")

    if incident.get("speed", 0) > 60:
        suggestions.append("Implement speed regulation enforcement.")

    if any("overlap" in c.lower() for c in incident.get("root_causes", [])):
        suggestions.append("Improve lane discipline enforcement.")

    suggestions.append("Deploy smart warning systems at intersection.")

    return suggestions
# ============================================
# DATABASE INITIALIZATION
# ============================================

db_conn = sqlite3.connect("incidents.db", check_same_thread=False)
db_cursor = db_conn.cursor()

db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        type TEXT,
        risk_level TEXT,
        severity TEXT,
        lat REAL,
        lng REAL
    )
""")

db_conn.commit()
def simulate_dispatch(incident_id):
    dispatch_log[incident_id] = {
        "status": "RESPONDING",
        "eta_minutes": round(np.random.uniform(2, 8), 2)
    }
def time_based_escalation(current_time, window_seconds=60, threshold=3):
    recent = []

    for i in incident_logs:
        try:
            incident_time = datetime.datetime.fromisoformat(i["time"])
            if (current_time - incident_time).total_seconds() <= window_seconds:
                recent.append(i)
        except Exception:
            continue

    return len(recent) >= threshold
# ============================================
# SUMMARY
# ============================================
async def process_summary_batch():
    while True:
        await asyncio.sleep(BATCH_TIMEOUT)

        if len(summary_queue) == 0:
            continue

        batch = []

        while summary_queue and len(batch) < BATCH_SIZE:
            batch.append(summary_queue.popleft())

        try:
            prompt = "Generate professional traffic accident summaries...\n\n"

            for i, inc in enumerate(batch):
                prompt += f"""
Incident {i+1}:
Type: {inc['type']}
Risk Level: {inc['risk_level']}
Confidence: {inc['confidence']}
Root Causes: {inc['root_causes']}
Speed: {inc.get('speed', 0)}
---
"""

            response = None

            for attempt in range(2):
                try:
                    response = await asyncio.to_thread(
                        groq_client.chat.completions.create,
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are a professional traffic forensic AI analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4
                    )
                    break
                except Exception as e:
                    if attempt == 1:
                        raise e
                    await asyncio.sleep(1)

            text = response.choices[0].message.content
            summaries = text.split("Incident")

            for idx, inc in enumerate(batch):
                if idx + 1 < len(summaries):
                    inc["ai_explanation"] = summaries[idx + 1].strip()

                    if main_loop:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_incident(inc),
                            main_loop
                        )

        except Exception as e:
            for inc in batch:
                inc["ai_explanation"] = f"AI summary failed: {str(e)}"

                if main_loop:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_incident(inc),
                        main_loop
                    )
# ============================================
# DETAILED SUMMARY
# ============================================
def generate_detailed_summary(incident):
    """
    Generate extended 100–200 line human-readable report summary
    """

    lines = []

    lines.append("Incident Overview:")
    lines.append("--------------------------------------------------")
    lines.append(f"The incident occurred on {incident['time']}.")
    lines.append(f"The system classified the event as {incident['type'].upper()}.")

    lines.append("")
    lines.append("Risk & Severity Assessment:")
    lines.append("--------------------------------------------------")
    lines.append(f"Risk Level: {incident['risk_level']}")
    lines.append(f"Risk Score: {round(incident['risk_score'],2)}")
    lines.append(f"Severity Classification: {incident['severity']}")

    lines.append("")
    lines.append("Behavioral Observations:")
    lines.append("--------------------------------------------------")

    if incident.get("speed", 0) > 60:
        lines.append("The vehicle was traveling at a significantly high speed.")
    elif incident.get("speed", 0) > 40:
        lines.append("The vehicle was traveling at moderate speed.")
    else:
        lines.append("The vehicle speed appeared within controlled limits.")

    if incident.get("overlap", 0) > 0.5:
        lines.append("Detected high bounding box overlap, indicating potential physical collision.")

    if incident.get("density_count", 0) > 3:
        lines.append("Multiple incidents detected in the same geographical zone, increasing escalation probability.")

    lines.append("")
    lines.append("Root Cause Analysis:")
    lines.append("--------------------------------------------------")

    for cause in incident.get("root_causes", []):
        lines.append(f"- {cause}")

    lines.append("")
    lines.append("AI Forensic Interpretation:")
    lines.append("--------------------------------------------------")
    lines.append(incident.get("ai_explanation", "AI explanation not available."))

    lines.append("")
    lines.append("Operational Impact:")
    lines.append("--------------------------------------------------")

    if incident["risk_level"] in ["HIGH", "CRITICAL"]:
        lines.append("Immediate dispatch intervention recommended.")
        lines.append("Traffic disruption probability: HIGH.")
        lines.append("Public safety risk: Elevated.")
    else:
        lines.append("Situation monitored under standard protocol.")
        lines.append("No immediate large-scale disruption detected.")

    lines.append("")
    lines.append("Preventive & Corrective Recommendations:")
    lines.append("--------------------------------------------------")

    for suggestion in incident.get("prevention_suggestions", []):
        lines.append(f"- {suggestion}")

    lines.append("")
    lines.append("Conclusion:")
    lines.append("--------------------------------------------------")
    lines.append(
        "Based on automated vision analytics, behavioral tracking, "
        "and AI-assisted forensic reasoning, this incident was detected "
        "with high analytical confidence. Continued monitoring and "
        "preventive infrastructure improvements are advised."
    )

    # Expand to ~150 lines by repeating structured commentary intelligently
    # Do NOT artificially pad the report
    # DO NOT PAD
    return "\n".join(lines)
# ============================================
# DETAILED SUMMARY LLM GROQ
# ============================================
async def generate_forensic_summary_llm(incident):
    """
    Generate a 150-line structured forensic investigation summary using Groq
    """

    prompt = f"""
You are a senior traffic forensic investigation analyst.

Generate a highly detailed structured forensic report (approx. 120–180 lines).

Structure it exactly with these sections:

1. Executive Overview
2. Environmental Context
3. Vehicle Behavior Analysis
4. Collision Dynamics Interpretation
5. Risk Escalation Reasoning
6. AI Confidence Assessment
7. Operational Impact Evaluation
8. Preventive Strategy Recommendations
9. Strategic Urban Insights
10. Final Forensic Conclusion

Incident Data:
- Type: {incident['type']}
- Risk Level: {incident['risk_level']}
- Risk Score: {incident['risk_score']}
- Severity: {incident['severity']}
- Speed: {incident.get('speed', 0)}
- Overlap: {incident.get('overlap', 0)}
- Density Count: {incident.get('density_count', 0)}
- Root Causes: {incident.get('root_causes')}
- AI Explanation: {incident.get('ai_explanation')}

Use professional investigative tone.
Avoid repetition.
Make it readable for non-technical officials.
"""

    try:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior forensic traffic investigation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception:
        return "LLM_UNAVAILABLE"
# ============================================
# TIMELINE GRAPH
# ============================================
def generate_timeline_graph(incident):
    """
    Generates a simple incident timeline visualization
    """
    timestamps = []
    risk_levels = []

    for inc in incident_logs:
        try:
            t = datetime.datetime.fromisoformat(inc["time"])
            timestamps.append(t)
            risk_levels.append(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(inc["risk_level"])
            )
        except:
            continue

    if not timestamps:
        return None

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, risk_levels, marker='o')
    plt.yticks([0,1,2,3], ["LOW","MEDIUM","HIGH","CRITICAL"])
    plt.xticks(rotation=45)
    plt.title("Incident Risk Timeline")
    plt.tight_layout()

    filename = "timeline_graph.png"
    plt.savefig(filename)
    plt.close()

    return filename
# ============================================
# TIMELINE GRAPH
# ============================================
def generate_dispatch_timeline_graph(incident):
    """
    Generates a dispatch lifecycle bar chart
    """

    created_time = datetime.datetime.fromisoformat(incident["time"])
    dispatch_info = dispatch_log.get(incident["id"])

    if not dispatch_info:
        return None

    eta_minutes = dispatch_info.get("eta_minutes", 5)

    responding_time = created_time + datetime.timedelta(minutes=1)
    arrival_time = responding_time + datetime.timedelta(minutes=eta_minutes)

    times = [created_time, responding_time, arrival_time]
    labels = ["Incident Created", "Dispatch Responding", "Estimated Arrival"]

    plt.figure(figsize=(8, 2))
    plt.hlines(1, times[0], times[-1])
    plt.scatter(times, [1, 1, 1])

    for i, label in enumerate(labels):
        plt.text(times[i], 1.02, label, rotation=45)

    plt.yticks([])
    plt.title("Dispatch Timeline Lifecycle")
    plt.tight_layout()

    filename = f"dispatch_timeline_{incident['id']}.png"
    plt.savefig(filename)
    plt.close()

    return filename

def generate_risk_breakdown_graph(incident):
    """
    Shows contribution of risk factors
    """
    confidence_score = incident["confidence"] * 40
    overlap_score = incident["overlap"] * 30
    speed_score = incident.get("speed", 0) * 0.5
    dl_bonus = incident.get("dl_bonus", 0)

    labels = ["Confidence", "Overlap", "Speed", "DL Bonus"]
    values = [confidence_score, overlap_score, speed_score, dl_bonus]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title("Risk Score Breakdown")
    plt.ylabel("Score Contribution")
    plt.tight_layout()

    filename = f"risk_breakdown_{incident['id']}.png"
    plt.savefig(filename)
    plt.close()

    return filename
# ============================================
# REPLAY GIF
# ============================================
def generate_replay_gif(frames, x1, y1, x2, y2, risk_level, speed, incident_id):
    gif_path = f"{SAVE_PATH}/incident_{incident_id}.gif"

    gif_frames = []

    for frame in frames:
        frame_copy = frame.copy()

        # Highlight box
        cv2.rectangle(
            frame_copy,
            (x1 - 5, y1 - 5),
            (x2 + 5, y2 + 5),
            (0, 0, 255),
            5
        )

        cv2.putText(
            frame_copy,
            f"{risk_level} | {int(speed)} km/h",
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255),
            2
        )

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)

    imageio.mimsave(gif_path, gif_frames, duration=0.05)

    return f"/snapshots/{os.path.basename(gif_path)}"

def open_browser():
    webbrowser.open("http://127.0.0.1:8000/dashboard")

# ============================================
# FRAME GENERATOR
# ============================================
def generate_frames():
    global collision_heatmap, stopped_heatmap
    global previous_centers, frame_buffer, previous_frame_time

    cap = cv2.VideoCapture(VIDEO_PATH)
    # 🔥 Warm-up DL model
    dummy = torch.zeros(sequence_length, 3, 224, 224).to(device)
    dummy = dummy.unsqueeze(0)

    with torch.no_grad():
        model(dummy)
    prev_time = 0
    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS
    frame_index = 0
    last_results = None
    vehicle_classes = [2, 3, 5, 7]
    try:
        while True:

            frame_index += 1
            success, frame = cap.read()
            if not success or frame is None:
                break

            annotated = frame.copy()
            replay_buffer.append(frame.copy())
            # ---------- INIT HEATMAP ----------
            if collision_heatmap is None:
                collision_heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
                stopped_heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

            # ---------- FPS ----------
            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (1100, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # ---------- DETECTION ----------
            # ---------- DETECTION (Optimized Frame Skipping) ----------
            small_frame = cv2.resize(frame, (416, 234), interpolation=cv2.INTER_AREA)

            # Run detection every 2 frames
            if frame_index % 6 == 0 or last_results is None:
                last_results = detector.detect(small_frame)

            results = last_results

            boxes = [
                tuple(map(int, box))
                for box, cls in zip(results.boxes.xyxy, results.boxes.cls)
                if int(cls) in vehicle_classes
            ]

            positions = tracker.update(boxes)

            incidents = incident_detector.check_incidents(positions, boxes)
            # ---------- DL MODEL ----------
            frame_prediction = 0
            frame_confidence = 0

            if dl_enabled:
                tensor_frame = transform(frame).to(device)
                frame_buffer.append(tensor_frame)

                if len(frame_buffer) > sequence_length:
                    frame_buffer.pop(0)

                if len(frame_buffer) == sequence_length:
                    if frame_index % 8 == 0:  # Run DL every 4 frames
                        input_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = torch.softmax(output, dim=1)
                            frame_prediction = torch.argmax(prob, dim=1).item()
                            frame_confidence = prob[0][frame_prediction].item()

                    if frame_prediction == 1 and frame_confidence > confidence_threshold:
                        cv2.putText(
                            annotated,
                            f"DL ACCIDENT ({frame_confidence:.2f})",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )
            # ================================
            # INCIDENT OVERRIDE (Risk + Escalation + Heat)
            # ================================
            critical_detected = False

            for idx, (obj_id, (cx, cy)) in enumerate(positions.items()):

                if idx >= len(boxes):
                    continue

                x1, y1, x2, y2 = boxes[idx]
                # ---- SPEED ----
                speed = 0
                current_time = frame_index / FPS_ASSUMED  # frame-based time

                if obj_id not in object_motion:
                    object_motion[obj_id] = {
                        "prev_center": (cx, cy),
                        "prev_time": current_time,
                        "prev_speed": 0
                    }
                else:
                    prev_data = object_motion[obj_id]

                    px, py = prev_data["prev_center"]
                    prev_time = prev_data["prev_time"]

                    time_delta = current_time - prev_time

                    if time_delta > 0:
                        pixel_distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                        distance_meters = pixel_distance * METERS_PER_PIXEL
                        speed_mps = distance_meters / time_delta
                        raw_speed = speed_mps * 3.6

                        # Smooth speed (EMA)
                        speed = 0.8 * prev_data["prev_speed"] + 0.2 * raw_speed

                        # Save
                        object_motion[obj_id]["prev_speed"] = speed
                        object_motion[obj_id]["prev_center"] = (cx, cy)
                        object_motion[obj_id]["prev_time"] = current_time
                # ---- ACCELERATION DETECTION ----
                prev_speed = previous_centers.get(f"{obj_id}_prev_speed", speed)
                acceleration = speed - prev_speed
                previous_centers[f"{obj_id}_prev_speed"] = speed
                # Store current time
                previous_centers[f"{obj_id}_time"] = current_time

                # Default box color
                color = (0, 255, 0)
                label = ""

                # If this object has an incident
                if obj_id in incidents:

                    data = incidents[obj_id]
                    overlap = data.get("debug_iou", 0)
                    # --- Impact spike detection ---
                    last_overlap = previous_centers.get(f"{obj_id}_last_overlap", 0)
                    impact_delta = overlap - last_overlap
                    previous_centers[f"{obj_id}_last_overlap"] = overlap
                    # ---------------------------
                    # SIMPLE COLLISION PERSISTENCE
                    # ---------------------------
                    if obj_id not in collision_persistence:
                        collision_persistence[obj_id] = 0

                    if overlap > 0.3:
                        collision_persistence[obj_id] += 1
                    else:
                        collision_persistence[obj_id] = 0

                    frames = collision_persistence[obj_id]

                    # ---------------------------
                    # SIMPLE RISK LOGIC
                    # ---------------------------
                    if overlap > 0.9:
                        base_level = "CRITICAL"

                    elif overlap > 0.8 and frames >= 3:
                        base_level = "CRITICAL"

                    elif impact_delta > 0.4:  # sudden overlap jump
                        base_level = "CRITICAL"

                    elif overlap > 0.7 and frames >= 2:
                        base_level = "HIGH"

                    elif overlap > 0.5:
                        base_level = "HIGH"

                    elif overlap > 0.3:
                        base_level = "MEDIUM"

                    else:
                        base_level = "LOW"
                    # ---- Sudden braking override ----
                    if acceleration < -20:
                        base_level = "CRITICAL"
                    # Require persistence for CRITICAL
                    if base_level == "CRITICAL" and frames < 3:
                        base_level = "HIGH"

                    risk_level = base_level
                    # ---------------------------
                    # Progressive Escalation
                    # ---------------------------
                    dl_bonus = 20 if (
                            frame_prediction == 1 and
                            frame_confidence > confidence_threshold
                    ) else 0
                    # 🔹 Define zone FIRST
                    global current_zone_id
                    # Get current zone
                    zone = ZONES[current_zone_id]
                    risk_score = (
                            float(data.get("confidence", 0)) * 60 +
                            overlap * 80 +
                            min(speed, 120) * 1.2 +  # speed cap
                            dl_bonus
                    )

                    # 🔥 ZONE-BASED TRAFFIC SCALING
                    traffic_type = zone["traffic"]

                    if traffic_type == "HIGH":
                        risk_score *= 1.2
                    elif traffic_type == "LOW":
                        risk_score *= 0.8

                    risk_score = min(risk_score, 200)
                    # ---------------------------
                    # ACCIDENT PROBABILITY SCORE
                    # ---------------------------
                    probability = min(100, int(
                        overlap * 60 +
                        min(speed, 90) * 0.5 +
                        max(0, -acceleration) * 1.2 +
                        dl_bonus
                    ))

                    # ---------------------------
                    # ROOT CAUSE ANALYSIS
                    # ---------------------------
                    root_causes = analyze_root_cause(
                        data,
                        speed,
                        overlap,
                        data.get("stopped_time", 0)
                    )

                    current_time = time.time()

                    if obj_id not in incident_cooldown or \
                            current_time - incident_cooldown[obj_id] > COOLDOWN_SECONDS:

                        # Update density ONLY when logging
                        density_count = update_density(zone["lat"], zone["lng"])

                        final_risk_level = risk_level
                        if density_count >= 3:
                            final_risk_level = "CRITICAL"
                            critical_detected = True

                        # Color resolution
                        if final_risk_level == "CRITICAL":
                            pulse = abs(math.sin(time.time() * 5))
                            intensity = int(200 + pulse * 55)
                            color = (0, 0, intensity)
                        elif final_risk_level == "HIGH":
                            color = (0, 120, 255)
                        elif final_risk_level == "MEDIUM":
                            color = (0, 165, 255)
                        else:
                            color = (0, 255, 0)

                        label = f"{data.get('type', '').upper()} | {final_risk_level}"

                        incident_id = len(incident_logs) + 1
                        timestamp = datetime.datetime.now().isoformat()

                        # Rotate zone for next incident
                        current_zone_id += 1
                        if current_zone_id > len(ZONES):
                            current_zone_id = 1

                        incident_data = {
                            "id": incident_id,
                            "time": timestamp,
                            "type": data.get("type"),
                            "risk_level": final_risk_level,
                            "risk_score": float(risk_score),
                            "severity": calculate_severity(
                                data.get("confidence", 0),
                                len(boxes)

                            ),
                            "confidence": float(data.get("confidence", 0)),
                            "overlap": float(overlap),
                            "speed": float(speed),
                            "acceleration": float(acceleration),
                            "accident_probability": probability,
                            "dl_bonus": dl_bonus,
                            "density_count": density_count,
                            "root_causes": root_causes,
                            "ai_explanation": generate_ai_explanation({
                                "risk_level": final_risk_level,
                                "confidence": data.get("confidence", 0),
                                "speed": speed,
                                "root_causes": root_causes
                            }),
                            "prevention_suggestions": generate_prevention_suggestions({
                                "risk_level": final_risk_level,
                                "speed": speed,
                                "root_causes": root_causes
                            }),
                            "zone_id": current_zone_id,
                            "zone_name": zone["name"],
                            "lat": zone["lat"],
                            "lng": zone["lng"],
                            "traffic_level": zone["traffic"],
                            "status": "ACTIVE"

                        }
                        # Rotate zone for next incident
                        current_zone_id += 1
                        if current_zone_id > len(ZONES):
                            current_zone_id = 1
                        # SAVE SNAPSHOT
                        snapshot_frame = frame.copy()
                        cv2.rectangle(snapshot_frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 6)
                        cv2.putText(snapshot_frame,
                                    f"{final_risk_level} | {int(speed)} km/h",
                                    (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    1,
                                    (0, 0, 255),
                                    3)

                        snapshot_filename = f"{SAVE_PATH}/incident_{incident_id}.jpg"
                        cv2.imwrite(snapshot_filename, snapshot_frame)
                        incident_data["image"] = f"/snapshots/{os.path.basename(snapshot_filename)}"
                        # ---------- GENERATE REPLAY GIF ----------
                        if len(replay_buffer) > 10:
                            threading.Thread(
                                target=generate_replay_gif,
                                args=(
                                    list(replay_buffer),
                                    x1, y1, x2, y2,
                                    final_risk_level,
                                    speed,
                                    incident_id
                                ),
                                daemon=True
                            ).start()
                        # STORE (ONLY ONCE)
                        incident_logs.append(incident_data)
                        summary_queue.append(incident_data)
                        incident_cooldown[obj_id] = current_time

                        # BROADCAST
                        if main_loop:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_incident(incident_data),
                                main_loop
                            )
                    # Heat blob
                    cv2.circle(
                        collision_heatmap if data.get("type") == "collision" else stopped_heatmap,
                        (int(cx), int(cy)),
                        50,
                        min(255, 100 + frames * 15),
                        -1
                    )

                # ---- DRAW BOX ----
                thickness = 2
                if label:
                    thickness = 4
                # Glow effect
                glow_color = color
                cv2.rectangle(
                    annotated,
                    (x1 - 3, y1 - 3),
                    (x2 + 3, y2 + 3),
                    glow_color,
                    6
                )
                cv2.rectangle(
                    annotated,
                    (x1 - 6, y1 - 6),
                    (x2 + 6, y2 + 6),
                    color,
                    8
                )

                if label:
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.2,
                        (0, 0, 255),
                        4
                    )

                # ---- SPEED LABEL (ALWAYS VISIBLE) ----
                speed_text = f"{int(speed)} km/h"

                cv2.putText(
                    annotated,
                    speed_text,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # ---------- HIGH ALERT (Calmer Animation) ----------
            if not critical_detected:

                # Check recent incidents
                recent_high = any(
                    log["risk_level"] == "HIGH"
                    for log in incident_logs[-3:]
                )

                if recent_high:
                    pulse = abs(math.sin(time.time() * 3))  # slow pulse
                    orange_intensity = int(150 + pulse * 50)

                    # Soft orange animated top bar
                    cv2.rectangle(
                        annotated,
                        (0, 0),
                        (1280, 60),
                        (0, orange_intensity, 255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        "HIGH RISK INCIDENT",
                        (420, 40),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
            # ---------- CRITICAL BANNER ----------
            if critical_detected:

                pulse = abs(math.sin(time.time() * 6))

                # Police-style red/blue flashing
                if int(time.time() * 4) % 2 == 0:
                    alert_color = (0, 0, 255)  # Red
                else:
                    alert_color = (255, 0, 0)  # Blue

                # Flashing top bar
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (1280, 90),
                    alert_color,
                    -1
                )

                # Glowing animated border
                border_thickness = int(8 + pulse * 6)

                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (1279, 719),
                    alert_color,
                    border_thickness
                )

                # Text shadow
                cv2.putText(
                    annotated,
                    "!!! CRITICAL INCIDENT DETECTED !!!",
                    (260, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    (0, 0, 0),
                    6
                )

                # Main glowing text
                cv2.putText(
                    annotated,
                    "!!! CRITICAL INCIDENT DETECTED !!!",
                    (260, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )

            # ---------- LIVE RISK METER ----------
            max_risk_score = 0
            for inc in incident_logs[-3:]:
                score = inc.get("risk_score", 0)

                if score > max_risk_score:
                    max_risk_score = score

            # Normalize to 0–100
            risk_percentage = min(int(max_risk_score), 150)
            risk_percentage = max(risk_percentage, 0)
            bar_width = int((risk_percentage / 150) * 300)

            # Background
            cv2.rectangle(annotated, (30, 650), (350, 680), (50, 50, 50), -1)

            # Fill
            if risk_percentage > 140:
                bar_color = (0, 0, 255)
            elif risk_percentage >= 95:
                bar_color = (0, 120, 255)
            else:
                bar_color = (0, 255, 0)

            cv2.rectangle(annotated, (30, 650), (30 + bar_width, 680), bar_color, -1)

            cv2.putText(
                annotated,
                f"Risk Level: {risk_percentage}",
                (30, 640),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            # ---------- HEATMAP OVERLAY (ALWAYS RUNS) ----------
            if heatmap_enabled:
                collision_heatmap *= heatmap_decay
                stopped_heatmap *= heatmap_decay

                combined = collision_heatmap + stopped_heatmap
                normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
                colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

                annotated = cv2.addWeighted(colored, 0.4, annotated, 0.6, 0)

                if np.max(combined) > 10:
                    cv2.putText(
                        annotated,
                        "HIGH RISK ZONE",
                        (900, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

            # ---------- UPDATE PREVIOUS CENTERS ----------
            for obj_id in positions:
                previous_centers[obj_id] = positions[obj_id]
            # ---------- CAMERA SHAKE FOR CRITICAL ----------
            if critical_detected:
                shake_intensity = 10  # adjust if needed
                dx = int(math.sin(time.time() * 15) * shake_intensity)
                dy = int(math.cos(time.time() * 12) * shake_intensity)

                h, w = annotated.shape[:2]

                # Create translation matrix
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                annotated = cv2.warpAffine(annotated, M, (w, h))
            annotated = cv2.resize(annotated, (1280, 720))
            # ---------- SYSTEM MONITOR ----------
            cpu_usage = psutil.cpu_percent()

            cv2.putText(
                annotated,
                f"CPU: {cpu_usage:.1f}%",
                (1100, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            # ---------- STREAM ----------
            loop_end = time.time()
            processing_time = loop_end - now  # now was defined earlier

            if processing_time < FRAME_TIME:
                time.sleep(FRAME_TIME - processing_time)

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

    finally:
        cap.release()
# ============================================
# WEBSOCKET
# ============================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_incident(incident):
    message = {
        "type": "incident",
        "incident": incident,
        "stats": {
            "total": len(incident_logs),
            "collisions": sum(1 for i in incident_logs if i["type"] == "collision"),
            "stopped": sum(1 for i in incident_logs if i["type"] == "stopped"),
        }
    }

    for conn in active_connections.copy():
        try:
            await conn.send_json(message)
        except Exception:
            if conn in active_connections:
                active_connections.remove(conn)
# ============================================
# ROUTES
# ============================================

@app.get("/")
def root():
    return {"status":"AID System Running"}

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/incidents")
def get_incidents():
    return incident_logs

@app.get("/stats")
def stats():
    total = len(incident_logs)
    collisions = sum(1 for i in incident_logs if i["type"]=="collision")
    stopped = sum(1 for i in incident_logs if i["type"]=="stopped")
    precision = collisions/(total+1e-5)

    return {
        "total_incidents": total,
        "collisions": collisions,
        "stopped": stopped,
        "precision": precision
    }

@app.get("/analytics")
def analytics():
    if collision_heatmap is None:
        return {"total_incidents": len(incident_logs), "heatmap_intensity": 0}

    return {
        "total_incidents": len(incident_logs),
        "heatmap_intensity": float(np.max(collision_heatmap + stopped_heatmap))
    }

@app.get("/export")
def export_csv():
    filename = "incidents.csv"

    if not incident_logs:
        return {"error": "No incidents to export"}

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=incident_logs[0].keys())
        writer.writeheader()
        writer.writerows(incident_logs)

    return FileResponse(
        path=filename,
        media_type="text/csv",
        filename="incidents.csv"
    )

@app.get("/save-heatmap")
def save_heatmap():
    if collision_heatmap is None:
        return {"error":"Heatmap not initialized"}

    combined = collision_heatmap + stopped_heatmap
    normalized = cv2.normalize(combined,None,0,255,cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8),cv2.COLORMAP_JET)
    filename="heatmap_snapshot.jpg"
    cv2.imwrite(filename,colored)
    return FileResponse(filename,media_type="image/jpeg")

@app.post("/set-threshold")
def set_threshold(value: float = Body(...)):
    global confidence_threshold
    confidence_threshold = value
    return {"threshold": confidence_threshold}

@app.post("/toggle-heatmap")
def toggle_heatmap():
    global heatmap_enabled
    heatmap_enabled = not heatmap_enabled
    return {"heatmap_enabled": heatmap_enabled}

@app.post("/toggle-dl")
def toggle_dl():
    global dl_enabled
    dl_enabled = not dl_enabled
    return {"dl_enabled": dl_enabled}

@app.post("/resolve/{index}")
def resolve_incident(index:int):
    if 0 <= index < len(incident_logs):
        incident_logs[index]["status"]="RESOLVED"
        return {"status":"resolved"}
    return {"error":"invalid index"}
@app.get("/report/{index}")
async def get_report(index: int):

    incident = incident_logs[index]

    forensic_summary = None

    try:
        forensic_summary = await generate_forensic_summary_llm(incident)
    except Exception:
        forensic_summary = None

    # CLEAN fallback logic
    if (
        not forensic_summary
        or "rate limit" in str(forensic_summary).lower()
        or "error" in str(forensic_summary).lower()
        or len(str(forensic_summary).strip()) < 200
    ):
        forensic_summary = generate_detailed_summary(incident)

    incident["forensic_summary"] = forensic_summary

    file = generate_incident_report(incident)
    return FileResponse(file, media_type="application/pdf")
@app.get("/hotspots")
def detect_hotspots():

    if len(incident_logs) < 3:
        return {"hotspots": []}

    coords = np.array([[i["lat"], i["lng"]] for i in incident_logs])

    clustering = DBSCAN(eps=0.002, min_samples=3).fit(coords)
    labels = clustering.labels_

    hotspots = []

    for label in set(labels):
        if label == -1:
            continue

        cluster_points = coords[labels == label]
        center = cluster_points.mean(axis=0)
        hotspots.append({
            "lat": float(center[0]),
            "lng": float(center[1]),
            "count": len(cluster_points)
        })

    return {"hotspots": hotspots}
@app.get("/dispatch/{incident_id}")
def get_dispatch(incident_id: int):
    return dispatch_log.get(incident_id, {"error": "Not found"})
@app.get("/intelligence")
async def analytics_intelligence():
    if not incident_logs:
        return {"message": "No incidents recorded."}

    total = len(incident_logs)
    critical = sum(1 for i in incident_logs if i["risk_level"] == "CRITICAL")
    high = sum(1 for i in incident_logs if i["risk_level"] == "HIGH")

    hotspot_zones = len(detect_hotspots()["hotspots"])

    trend = "Stable"
    if critical > 5:
        trend = "Escalating Risk Trend"
    elif high > 5:
        trend = "Moderate Risk Growth"

    return {
        "total_incidents": total,
        "critical_count": critical,
        "high_count": high,
        "hotspot_zones": hotspot_zones,
        "risk_trend": trend,
        "intelligence_summary": (
            "Automated analytics indicate spatial clustering and "
            "risk escalation patterns. Monitoring intensity adjustment recommended."
        )
    }
@app.get("/dashboard")
def dashboard():
    return FileResponse("frontend/index.html")
@app.get("/city-analytics")
def city_analytics():

    if not incident_logs:
        return {
            "zones": [],
            "city_risk_index": 0,
            "most_critical_zone": None
        }

    zone_stats = {}

    for inc in incident_logs:
        zone = inc.get("zone_name", "Unknown")

        if zone not in zone_stats:
            zone_stats[zone] = {
                "total": 0,
                "critical": 0,
                "high": 0
            }

        zone_stats[zone]["total"] += 1

        if inc["risk_level"] == "CRITICAL":
            zone_stats[zone]["critical"] += 1
        elif inc["risk_level"] == "HIGH":
            zone_stats[zone]["high"] += 1

    # Compute City Risk Index
    recent = incident_logs[-10:]
    city_risk_index = 0

    if recent:
        city_risk_index = sum(i["risk_score"] for i in recent) / len(recent)

    # Find most critical zone
    most_critical_zone = None
    max_critical = 0

    for zone, stats in zone_stats.items():
        if stats["critical"] > max_critical:
            max_critical = stats["critical"]
            most_critical_zone = zone

    return {
        "zones": zone_stats,
        "city_risk_index": round(city_risk_index, 2),
        "most_critical_zone": most_critical_zone
    }
@app.get("/predict-escalation")
def predict_escalation():

    predictions = {}

    for zone_id, zone in ZONES.items():

        zone_name = zone["name"]

        recent = [
            inc for inc in incident_logs[-10:]
            if inc["zone_name"] == zone_name
        ]

        if not recent:
            predictions[zone_name] = 0
            continue

        avg_risk = sum(i["risk_score"] for i in recent) / len(recent)

        escalation_probability = min(100, int(avg_risk / 2))

        predictions[zone_name] = escalation_probability

    return {"predictions": predictions}
@app.get("/zone-leaderboard")
def zone_leaderboard():

    zone_scores = {}

    for inc in incident_logs:
        zone = inc.get("zone_name", "Unknown")

        if zone not in zone_scores:
            zone_scores[zone] = 0

        zone_scores[zone] += inc.get("risk_score", 0)

    sorted_zones = sorted(
        zone_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {"leaderboard": sorted_zones}
@app.get("/zone-heat")
def zone_heat():

    zone_heat = {}

    for inc in incident_logs:
        zone = inc.get("zone_name", "Unknown")

        if zone not in zone_heat:
            zone_heat[zone] = 0

        # Weighted heat
        if inc["risk_level"] == "CRITICAL":
            zone_heat[zone] += 5
        elif inc["risk_level"] == "HIGH":
            zone_heat[zone] += 3
        elif inc["risk_level"] == "MEDIUM":
            zone_heat[zone] += 1

    return {"zone_heat": zone_heat}
import os
import cv2
import time
import serial
import pickle
import numpy as np
from typing import Annotated, TypedDict, List
from sklearn.linear_model import LinearRegression

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# --- State Schema ---
class TrafficState(TypedDict):
    messages: Annotated[list, add_messages]
    vehicle_counts: List[int]
    estimated_commute_time: float
    video_paths: List[str]

# --- Load or Initialize Learning Model ---
MODEL_PATH = "commute_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = LinearRegression()
    model.fit(np.array([[0], [10], [20], [30], [40]]), np.array([10, 12, 15, 20, 25]))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# --- Agents ---
def analyze_traffic(state: TrafficState) -> dict:
    counts = []
    for path in state["video_paths"]:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if not ret:
            counts.append(0)
            continue
        fg_mask = cv2.createBackgroundSubtractorMOG2().apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicle_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
        counts.append(vehicle_count)
        cap.release()
    print("Vehicle counts:", counts)
    return {"vehicle_counts": counts}

def predict_congestion(state: TrafficState) -> dict:
    total = sum(state["vehicle_counts"])
    congestion = min(total / 40.0, 1.0)
    return {"congestion_level": congestion}

def estimate_commute_time(state: TrafficState) -> dict:
    count_sum = sum(state["vehicle_counts"])
    X = np.array([[count_sum]])
    predicted = float(model.predict(X)[0])
    print(f"Predicted commute time: {predicted:.2f} minutes")
    return {"estimated_commute_time": predicted}

def control_lights(state: TrafficState) -> dict:
    try:
        ser = serial.Serial("COM4", 9600, timeout=1)
        max_lane = state["vehicle_counts"].index(max(state["vehicle_counts"]))
        command = f"P{max_lane+1}T6\n"
        ser.write(command.encode())
        ser.close()
        print(f"Sent to Arduino: {command.strip()}")
    except Exception as e:
        print(f"[Warning] Serial issue: {e}")
    return {}

def learn_from_data(state: TrafficState) -> dict:
    try:
        X_new = np.array([[sum(state["vehicle_counts"])]])
        y_new = np.array([state["estimated_commute_time"]])
        model.partial_fit(X_new, y_new)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Learning updated with: X={X_new[0][0]}, y={y_new[0]:.2f}")
    except Exception as e:
        print(f"[Error] Learning failed: {e}")
    return {}

# --- Graph Setup ---
graph_builder = StateGraph(TrafficState)
graph_builder.add_node("analyze", analyze_traffic)
graph_builder.add_node("predict", predict_congestion)
graph_builder.add_node("estimate", estimate_commute_time)
graph_builder.add_node("control", control_lights)
graph_builder.add_node("learn", learn_from_data)

graph_builder.set_entry_point("analyze")
graph_builder.add_edge("analyze", "predict")
graph_builder.add_edge("predict", "estimate")
graph_builder.add_edge("estimate", "control")
graph_builder.add_edge("control", "learn")
graph_builder.add_edge("learn", "analyze")

# --- Persistent Memory ---
graph = graph_builder.compile()

# --- Start Execution ---
if __name__ == "__main__":
    initial_state = {
        "messages": [],
        "vehicle_counts": [],
        "estimated_commute_time": 0.0,
        "video_paths": [
            r"C:\Users\user\Desktop\5th\lane1.mp4",
            r"C:\Users\user\Desktop\5th\lane2.mp4",
            r"C:\Users\user\Desktop\5th\lane3.mp4",
            r"C:\Users\user\Desktop\5th\lane4.mp4",
        ]
    }

    config = {"configurable": {"thread_id": "traffic-cycle"}}

    print("🚦 Starting LangGraph Traffic Agent...")
    while True:
        events = graph.stream(initial_state, config, stream_mode="values")
        for event in events:
            pass
        time.sleep(5)  # slight pause to avoid overprocessing

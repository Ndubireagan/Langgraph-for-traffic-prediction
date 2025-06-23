# 🚦 LangGraph Traffic AI Agent

This project implements a **real-time intelligent traffic control system** using [LangGraph](https://github.com/langchain-ai/langgraph), OpenCV, and a lightweight machine learning model. The system analyzes traffic from video feeds, predicts congestion, estimates commute time between two points, and dynamically controls traffic lights via Arduino.

---

## 📌 Features

- 🔁 **Cyclic AI agent** using LangGraph to continuously analyze and react.
- 🎥 **Video-based vehicle detection** from 4 lanes.
- 📊 **ML-based commute time prediction** using real-time vehicle counts.
- 🤖 **Arduino traffic light control** via serial communication.
- 🧠 **Learning system** that improves commute estimation over time.
- 💾 **Persistent state memory** using SQLite.

---

## 🧠 How It Works

The agent cycles through the following tasks:

1. **Analyze Traffic (`analyze`)**
   - Processes each video feed.
   - Counts vehicles using OpenCV and background subtraction.

2. **Predict Congestion (`predict`)**
   - Calculates a normalized congestion score based on vehicle count.

3. **Estimate Commute Time (`estimate`)**
   - Uses a scikit-learn regression model to estimate travel time between Point A and B.

4. **Control Traffic Lights (`control`)**
   - Sends serial commands to an Arduino to activate the green light for the busiest lane.

5. **Learn From Data (`learn`)**
   - Updates the ML model with the most recent traffic and commute data to improve future predictions.

6. **Cycle**
   - The loop starts again, creating an autonomous real-time system.

---

## 📁 Main File: `traffic_ai_agent.py`

This script contains:

- 📦 **State Management**: `TrafficState` tracks counts, video paths, and commute times.
- 🧠 **Agent Nodes**: Defined for analysis, prediction, estimation, control, and learning.
- 🔄 **LangGraph Workflow**: Configured as a cyclic graph with persistent memory.
- 🔌 **Execution Loop**: Continuously runs with optional sleep delay.

---

## 🚀 Getting Started

### ✅ Install Dependencies

```bash
pip install langgraph opencv-python scikit-learn pyserial

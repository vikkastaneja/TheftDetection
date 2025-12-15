# Package Theft Detection System - Walkthrough

## How to Run
1.  **(Optional) Create Virtual Environment:**
    `python -m venv myenv`
2.  **(Optional) Activate Virtual Environment:**
    `source ./myvenv/bin/activate`
3.  **Install required files:**
    `pip install -r requirements.txt`
4.  **Set API Key:**
    Ensure `TheftDetection/.env` contains your `OPENAI_API_KEY`.
5.  **Run:**
    `python main.py`

## User Guide
### 1. The Interface
*   **Video Feed**: Shows live camera stream.
*   **Tracking**: Shows "Tracking: X pkgs" at the top.
*   **Faces**: Shows green labels for known people (e.g., "Owner", "Resident_Auto_1").

### 2. Auto-Learning (The "Magic")
*   **The Problem**: Teaching the system who is safe without manual entry.
*   **The Solution**: Just stand in front of the camera (close enough to fill 10% of the view) for **2-3 seconds**.
*   **Confirmation**: Console prints `[AUTO-LEARN] Trusting new face: Resident_Auto_1`. You are now trusted.

### 3. Theft Scenarios
#### Scenario A: Authorized Pickup (You)
*   **Action**: You (a known face) pick up a package.
*   **Result**: 
    *   System detects "Package Taken by OWNER".
    *   **NO ALARM**.
    *   **NO AI CALL** (Saves money).

#### Scenario B: Theft (Stranger)
*   **Action**: Someone unknown (or you hiding your face) picks it up.
*   **Result**: 
    *   System detects "Package Taken (Potential Theft!)".
    *   **ALERT 1**: "POTENTIAL THEFT" logged to `alerts.log`.
    *   **AI AGENT**: System sends video clip to OpenAI.
    *   **VERDICT**: AI analyzes clip. If it confirms theft -> **"THEFT CONFIRMED"** alert.

## Files of Interest
*   `alerts.log`: Text file history of all detection events.
*   `faces/`: Folder containing images of everyone the system has learned. You can delete images here to "reset" a person.

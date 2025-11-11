import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from tkinter import Tk, filedialog
import warnings
import os # --- NEW: Import os for file path handling

# Suppress some specific warnings from insightface
warnings.filterwarnings("ignore", category=UserWarning, module='insightface.model_zoo')
warnings.filterwarnings("ignore", category=FutureWarning, module='insightface.model_zoo')

# ----------------------------------------------------
# --- File Selection Logic (Unchanged) ---
# ----------------------------------------------------
print("Opening file dialog to select a PHOTO OF A FACE...")
root = Tk()
root.withdraw()
source_path = filedialog.askopenfilename(
    title="Select a photo of a face (PNG or JPG)",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)
root.destroy()

if not source_path:
    print("No file selected. Exiting application.")
    exit()
print(f"File selected: {source_path}")

# ----------------------------------------------------
# --- MODIFIED: Step 1 - Load Models ---
# ----------------------------------------------------
print("Loading InsightFace models...")

# --- This part is UNCHANGED ---
# Initialize FaceAnalysis to find faces
# This will use its own cached models (or try to download them
# to its default location ~/.insightface/models)
app = FaceAnalysis(providers=['DmlExecutionProvider']) 
app.prepare(ctx_id=0, det_size=(640, 640))

# --- This part is MODIFIED ---
# Define the local path to your swapper model
model_folder = 'models'
swapper_file = 'inswapper_128.onnx'
swapper_path = os.path.join(model_folder, swapper_file)

# Check if the file actually exists
if not os.path.exists(swapper_path):
    print(f"Error: Model not found at {swapper_path}")
    print(f"Please make sure '{swapper_file}' is inside a folder named '{model_folder}'.")
    exit()

print(f"Loading local swapper model from: {swapper_path}")

# Initialize the Face Swapper model from the LOCAL FILE
# Initialize the Face Swapper model from the LOCAL FILE
try:
    swapper = insightface.model_zoo.get_model(
        swapper_path, 
        providers=['DmlExecutionProvider'] # --- MODIFIED: Using DirectML (AMD GPU) ---
    )
    print("Face swapper model loaded successfully.")
except Exception as e:
    print(f"Error loading swapper model: {e}")
    print("The model file might be corrupt, or you may have an onnxruntime mismatch.")
    exit()

# ----------------------------------------------------
# --- Step 2: Analyze Source Face (Unchanged) ---
# ----------------------------------------------------
source_img = cv2.imread(source_path)
if source_img is None:
    print("Error: Could not load source image.")
    exit()

source_faces = app.get(source_img)

if not source_faces:
    print("Error: No face found in the selected source image.")
    exit()

source_face = source_faces[0]
print("Source face analyzed successfully.")

# ----------------------------------------------------
# --- Step 3: Webcam Loop (Unchanged) ---
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    try:
        target_faces = app.get(frame)

        if target_faces:
            output_frame = frame.copy()
            
            for target_face in target_faces:
                output_frame = swapper.get(
                    output_frame,
                    target_face,
                    source_face,
                    paste_back=True
                )
            
            cv2.imshow('AI Face Swapper', output_frame)

        else:
            cv2.imshow('AI Face Swapper', frame)

    except Exception as e:
        print(f"Error during swap: {e}")
        cv2.imshow('AI Face Swapper', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
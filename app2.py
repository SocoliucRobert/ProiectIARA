import sys
import os

# --- NEW FIX: Redirect stdout/stderr to a log file ---
# This must be at the very top, before any other library is used
# This will stop the 'tqdm' progress bar from crashing the app
#
# Check if we are running as a PyInstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # We are in a bundle. Get the path to the .exe's folder
    exe_path = os.path.dirname(sys.executable)
    log_file_path = os.path.join(exe_path, 'app.log')
else:
    # We are running as a normal .py script
    log_file_path = 'app.log'

# Redirect stdout and stderr to the log file
try:
    # Open the log file in 'w' (write) mode, with utf-8 encoding
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file
except Exception as e:
    # This is a last resort print, in case log file opening fails
    print(f"FATAL: Failed to open log file: {e}") 
# --- END NEW FIX ---


import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from tkinter import Tk, filedialog
import warnings
# 'os' and 'sys' are already imported above


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # Check if we are running in a PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        # Not in a bundle, so we are in normal Python dev
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)

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
    sys.exit()
print(f"File selected: {source_path}")

# ----------------------------------------------------
# --- MODIFIED: Step 1 - Load Models ---
# ----------------------------------------------------
print("Loading InsightFace models...")

model_dir = resource_path('insightface')
os.environ['INSIGHTFACE_HOME'] = model_dir
print(f"Set INSIGHTFACE_HOME to: {model_dir}")


# Initialize FaceAnalysis to find faces
# We are keeping the root=model_dir fix
try:
    app = FaceAnalysis(root=model_dir, providers=['DmlExecutionProvider']) 
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("FaceAnalysis app prepared successfully.")
except Exception as e:
    print(f"--- CRITICAL ERROR during FaceAnalysis init ---")
    print(str(e))
    # This will write the full traceback to the log
    import traceback
    traceback.print_exc()
    print("-------------------------------------------------")
    sys.exit()


# --- This part is MODIFIED ---
# Define the local path to your swapper model
swapper_file = 'inswapper_128.onnx' # Typo from user, should be inswapper

# Use our new helper function to get the TRUE path
relative_model_path = os.path.join('models', swapper_file)
swapper_path = resource_path(relative_model_path)

print(f"Attempting to load swapper model from: {swapper_path}")
# Check if the file actually exists
if not os.path.exists(swapper_path):
    print(f"Error: Model not found at {swapper_path}")
    print(f"Please make sure '{swapper_file}' is inside a folder named 'models'.")
    sys.exit()

print(f"Loading local swapper model from: {swapper_path}")

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
    import traceback
    traceback.print_exc()
    sys.exit()

# ----------------------------------------------------
# --- Step 2: Analyze Source Face (Unchanged) ---
# ----------------------------------------------------
print("Analyzing source face...")
source_img = cv2.imread(source_path)
if source_img is None:
    print("Error: Could not load source image.")
    sys.exit()

source_faces = app.get(source_img)

if not source_faces:
    print("Error: No face found in the selected source image.")
    sys.exit()

source_face = source_faces[0]
print("Source face analyzed successfully.")

# ----------------------------------------------------
# --- Step 3: Webcam Loop (Unchanged) ---
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("Webcam opened. Press 'q' to quit.")
cv2.namedWindow('MAREA DEGHIZARE', cv2.WINDOW_NORMAL)
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
            
            cv2.imshow('MAREA DEGHIZARE', output_frame)

        else:
            cv2.imshow('MAREA DEGHIZARE', frame)

    except Exception as e:
        print(f"Error during swap: {e}")
        cv2.imshow('MAREA DEGHIZARE', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("App shutting down.")
cap.release()
cv2.destroyAllWindows()
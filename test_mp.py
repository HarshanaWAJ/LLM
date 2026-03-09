import cv2, numpy as np, os

def extract_skeleton_from_cartoon_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Given a white-bg cartoon stick figure frame, return a dark-bg skeleton render."""
    h, w = frame_bgr.shape[:2]

    # Convert to grayscale and invert (dark lines → white on black)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold: lines are dark on white bg
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize using morphological thinning
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        temp_skel = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, temp_skel)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # Dilate skeleton slightly and color it cyan on black background
    skel_dilated = cv2.dilate(skel, element, iterations=2)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[skel_dilated > 0] = [0, 200, 255]  # BGR: cyan

    return canvas

# Test on one frame
cap = cv2.VideoCapture("data/animations/running fast.mp4")
for _ in range(10):
    ok, f = cap.read()
cap.release()

result = extract_skeleton_from_cartoon_frame(f)
os.makedirs("data/generated", exist_ok=True)
cv2.imwrite("data/generated/test_skel.png", result)
print("Saved test_skel.png. Shape:", result.shape)

import cv2

# Input: 96x96 colored pixel matrix of the screen (including the UI bar at bottom)
# Output: cropped normalized grayscale matrix (without the UI bar)
def process_image(state):
    state = state[0:84, :] # crop bottom UI (12 pixels)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0  # normalize grayscale value
    return state
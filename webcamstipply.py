
import cv2
import numpy as np

enable_sobel = 0
use_combined_edges = 0
gaussian_kernel_size = 5
edge_color = [255, 255, 255]

# Default parameters for edge styles
circle_edges = 0
stylization_mode = 0

low_threshold = 33
high_threshold = 64

def process_frame(frame, low_threshold, high_threshold, gaussian_kernel_size, enable_sobel, use_combined_edges, edge_color, circle_edges, stylization_mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred_gray = cv2.GaussianBlur(gray, (gaussian_kernel_size, gaussian_kernel_size), 0)

    edges_gray = cv2.Canny(blurred_gray, low_threshold, high_threshold)

    if use_combined_edges:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        blurred_s = cv2.GaussianBlur(s_channel, (gaussian_kernel_size, gaussian_kernel_size), 0)
        edges_s = cv2.Canny(blurred_s, low_threshold, high_threshold)
        combined_edges = cv2.bitwise_or(edges_gray, edges_s)
    else:
        combined_edges = edges_gray

    if enable_sobel:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=gaussian_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=gaussian_kernel_size)
        sobel_edges = cv2.bitwise_or(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))
        combined_edges = cv2.bitwise_or(combined_edges, sobel_edges)

    if stylization_mode == 1:
        combined_edges = cv2.GaussianBlur(combined_edges, (5, 5), 0)
    elif stylization_mode == 2:
        kernel = np.ones((3, 3), np.uint8)
        combined_edges = cv2.dilate(combined_edges, kernel, iterations=2)

    if circle_edges:
        kernel = np.ones((3, 3), np.uint8)
        combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
        combined_edges = cv2.erode(combined_edges, kernel, iterations=1)

    colored_edges = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
    colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = edge_color

    return colored_edges

def on_low_threshold_trackbar(val):
    global low_threshold
    low_threshold = val

def on_high_threshold_trackbar(val):
    global high_threshold
    high_threshold = val

def on_gaussian_kernel_size_trackbar(val):
    global gaussian_kernel_size
    gaussian_kernel_size = max(1, val | 1)

def on_sobel_trackbar(val):
    global enable_sobel
    enable_sobel = val

def on_combined_edges_trackbar(val):
    global use_combined_edges
    use_combined_edges = val

def on_red_trackbar(val):
    global edge_color
    edge_color[2] = val

def on_green_trackbar(val):
    global edge_color
    edge_color[1] = val

def on_blue_trackbar(val):
    global edge_color
    edge_color[0] = val

def on_circle_edges_trackbar(val):
    global circle_edges
    circle_edges = val

def on_stylization_mode_trackbar(val):
    global stylization_mode
    stylization_mode = val

# Start webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Edge Detection')
cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Controls', 700, 100)

# Create trackbars for customizing shizz
cv2.createTrackbar('Low Threshold', 'Controls', low_threshold, 255, on_low_threshold_trackbar)
cv2.createTrackbar('High Threshold', 'Controls', high_threshold, 255, on_high_threshold_trackbar)
cv2.createTrackbar('Gaussian Blur Kernel Size', 'Controls', gaussian_kernel_size, 19, on_gaussian_kernel_size_trackbar)
#cv2.createTrackbar('Enable Sobel', 'Controls', enable_sobel, 1, on_sobel_trackbar)
cv2.createTrackbar('Use Combined Edges', 'Controls', use_combined_edges, 1, on_combined_edges_trackbar)
cv2.createTrackbar('R', 'Controls', edge_color[2], 255, on_red_trackbar)
cv2.createTrackbar('G', 'Controls', edge_color[1], 255, on_green_trackbar)
cv2.createTrackbar('B', 'Controls', edge_color[0], 255, on_blue_trackbar)
cv2.createTrackbar('Circle Edges', 'Controls', circle_edges, 1, on_circle_edges_trackbar)
cv2.createTrackbar('Stylization Mode', 'Controls', stylization_mode, 10, on_stylization_mode_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame, low_threshold, high_threshold, gaussian_kernel_size, enable_sobel, use_combined_edges, edge_color, circle_edges, stylization_mode)
    cv2.imshow('Edge Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

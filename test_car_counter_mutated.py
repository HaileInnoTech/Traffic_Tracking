import cv2
import numpy as np
import pytest
from tracker import EuclideanDistTracker
from TrackingTrafficModule import process_frame  

@pytest.fixture
def setupMutated():
    object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    tracker = EuclideanDistTracker()
    pts = np.array([[370, 400], [750, 300], [820, 400], [300, 400], [370, 400]], np.int32)


    car_counter = 0
    seen_ids = set()
    return object_detector, tracker, pts, car_counter, seen_ids

def test_car_counter_originalMutated(setupMutated):
    object_detector, tracker, pts, car_counter, seen_ids = setupMutated
    video = cv2.VideoCapture("traffic.mp4")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame, mask, roi, car_counter, seen_ids = process_frame(frame, object_detector, tracker, pts, car_counter, seen_ids)
    video.release()
    print("Quantity of traffic =", car_counter)
    expected_car_count = car_counter
    assert car_counter == expected_car_count


def test_car_counter_skipFrameMutated(setupMutated):
    object_detector, tracker, pts, car_counter, seen_ids = setupMutated
    video = cv2.VideoCapture("traffic.mp4")
    frame_count = 0
    skip_frames = 2
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame, mask, roi, car_counter, seen_ids = process_frame(frame, object_detector, tracker, pts, car_counter,
                                                                    seen_ids)
        frame_count += 1
    video.release()
    expected_car_count = 154
    print(car_counter)
    assert car_counter == expected_car_count

def test_car_counter_originalReverseMutated(setupMutated):
    object_detector, tracker, pts, car_counter, seen_ids = setupMutated
    video = cv2.VideoCapture("trafficReverse.mp4")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame, mask, roi, car_counter, seen_ids = process_frame(frame, object_detector, tracker, pts, car_counter, seen_ids)
    video.release()
    expected_car_count = 154
    assert car_counter == expected_car_count

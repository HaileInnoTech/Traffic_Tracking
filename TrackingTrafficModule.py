import cv2
import numpy as np
from tracker import EuclideanDistTracker


def process_frame(frame, object_detector, tracker, pts, car_counter, seen_ids):
    roi = frame[400:500, 150:1000]  # Crop region of interest (ROI)
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    frame = cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    for cnt in contours:  # Find moving contour
        area = cv2.contourArea(cnt)
        if area > 1000:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            detections.append([x1, y1, w1, h1])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:  # Print and detect id if in an area
        x, y, w, h, id = box_id
        result = cv2.pointPolygonTest(pts, (x, y), False)
        if result is not None:
            cv2.putText(roi, str(id), (x, y - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(roi, (x, y), 5, color=(255, 0, 255), thickness=2)
        if id not in seen_ids:
            car_counter += 1
            seen_ids.add(id)

    cv2.putText(frame, f'Cars Detected: {car_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame, mask, roi, car_counter, seen_ids


def main():
    video = cv2.VideoCapture("traffic.mp4")
    object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=10)
    tracker = EuclideanDistTracker()

    # Polygon corner points coordinates
    pts = np.array([[370, 400], [750, 400], [820, 500],
                    [300, 500], [370, 400]], np.int32)
    car_counter = 0  # Initialize counter
    seen_ids = set()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame, mask, roi, car_counter, seen_ids = process_frame(frame, object_detector, tracker, pts, car_counter,
                                                                seen_ids)

        cv2.imshow("Mask Video", mask)
        cv2.imshow("Video", frame)
        cv2.imshow("Roi", roi)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

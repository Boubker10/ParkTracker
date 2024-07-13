import cv2
import pickle
import numpy as np
import argparse

# Charger la liste des positions de parking si elle existe
def load_parking_positions(filename='CarParkPos'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

# Sauvegarder la liste des positions de parking
def save_parking_positions(posList, filename='CarParkPos'):
    with open(filename, 'wb') as f:
        pickle.dump(posList, f)

def mouseClick(events, x, y, flags, params):
    posList, save_file = params
    if events == cv2.EVENT_LBUTTONDOWN:
        if len(posList) == 0 or len(posList[-1]) == 4:
            posList.append([])
        posList[-1].append((x, y))
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            for j, pos in enumerate(polygon):
                if abs(pos[0] - x) < 10 and abs(pos[1] - y) < 10:
                    posList[i].pop(j)
                    break
            if len(polygon) == 0:
                posList.pop(i)
                break
    save_parking_positions(posList, save_file)

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def transform_point(point, matrix):
    if isinstance(point, tuple) and len(point) == 2:
        px, py = point
        transformed_point = cv2.perspectiveTransform(np.array([[px, py]], dtype='float32').reshape(-1, 1, 2), matrix)
        return transformed_point[0][0]
    else:
        raise ValueError(f"Point {point} is not a valid (x, y) tuple")

def check_occupancy(frame, posList, detections, matrix):
    occupied_positions = []
    for polygon in posList:
        transformed_polygon = [transform_point(p, matrix) for p in polygon]
        for detection in detections:
            if int(detection[5]) == 2:  # 2 est l'index de la classe 'car'
                car_x1, car_y1, car_x2, car_y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
                car_polygon = [(car_x1, car_y1), (car_x2, car_y1), (car_x2, car_y2), (car_x1, car_y2)]
                if any(point_in_polygon(p, car_polygon) for p in transformed_polygon):
                    occupied_positions.append(polygon)
                    break
    return occupied_positions

def process_image(image_path, posList, matrix):
    frame = cv2.imread(image_path)
    detections = []  # Ajouter ici le code pour les détections
    occupied_positions = check_occupancy(frame, posList, detections, matrix)

    for polygon in posList:
        color = (0, 0, 255) if polygon in occupied_positions else (0, 255, 0)
        for i in range(len(polygon)):
            cv2.line(frame, polygon[i], polygon[(i+1) % len(polygon)], color, 2)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, posList, matrix):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouseClick, [posList, 'CarParkPos'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []  # Ajouter ici le code pour les détections
        occupied_positions = check_occupancy(frame, posList, detections, matrix)

        for polygon in posList:
            color = (0, 0, 255) if polygon in occupied_positions else (0, 255, 0)
            for i in range(len(polygon)):
                cv2.line(frame, polygon[i], polygon[(i+1) % len(polygon)], color, 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Slot Occupancy Detection")
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--video', type=str, help='Path to the video file')
    args = parser.parse_args()

    posList = load_parking_positions()

    # Matrice de transformation perspective (doit être définie selon vos besoins)
    src_points = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype='float32')
    dst_points = np.array([[20, 20], [200, 50], [220, 200], [20, 250]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    if args.image:
        process_image(args.image, posList, matrix)
    elif args.video:
        process_video(args.video, posList, matrix)
    else:
        print("Please provide either --image or --video argument.")

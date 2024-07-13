import argparse
import cv2
import pickle
import cvzone
import numpy as np
import torch

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Classes pour les voitures, camions et motos
classes_to_detect = [2, 5, 7]

def process_parking(video_path=None, image_paths=None, positions_path="CarParkPos"):
    # Charger les positions de parking précédemment enregistrées
    with open(positions_path, "rb") as f:
        pos_list = pickle.load(f)

    # Fonction pour vérifier les espaces de parking
    def check_parking_space(img, img_pro, contours):
        space_counter = 0

        for cnt in contours:
            # Approximer le contour pour obtenir un polygone
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Ignorer les petits contours ou ceux qui n'ont pas 4 côtés
            if len(approx) != 4 or cv2.contourArea(approx) < 500:
                continue

            # Déterminer les coordonnées du polygone
            poly = [point[0].tolist() for point in approx]

            # Convertir en tableau numpy
            pts = np.array(poly, np.int32)
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            cropped_img = img_pro[y:y+h, x:x+w].copy()
            pts = pts - pts.min(axis=0)

            mask = np.zeros(cropped_img.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
            count = cv2.countNonZero(mask)

            if count < 800:
                color = (0, 255, 0)
                thickness = 5
                space_counter += 1
            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.polylines(img, [np.array(poly, np.int32)], isClosed=True, color=color, thickness=thickness)

        cvzone.putTextRect(img, f"Free :- {space_counter}/{len(contours)}", (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

    # Fonction pour détecter les voitures avec YOLOv5
    def detect_cars(img):
        results = model(img)
        return results

    # Si un chemin de vidéo est fourni
    if video_path:
        cap = cv2.VideoCapture(video_path)
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = cap.read()
            if not success:
                break

            # Détecter les voitures
            results = detect_cars(img)
            detections = results.xyxy[0]

            for detection in detections:
                if int(detection[5]) in classes_to_detect:
                    x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[int(detection[5])]} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
            img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            img_median = cv2.medianBlur(img_threshold, 5)

            # Dilater l'image pour rendre les lignes blanches plus épaisses
            kernel = np.ones((3, 3), np.uint8)
            img_dilate = cv2.dilate(img_median, kernel, iterations=1)

            # Détection des contours
            contours, _ = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 2)

            # Vérifier les espaces de parking
            check_parking_space(img_contours, img_dilate, contours)

            # Afficher la vidéo image par image
            cv2.imshow("Image", img_contours)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Si des chemins d'images sont fournis
    elif image_paths:
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Détecter les voitures
            results = detect_cars(img)
            detections = results.xyxy[0]

            for detection in detections:
                if int(detection[5]) in classes_to_detect:
                    x1, y1, x2, y2, conf = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), detection[4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[int(detection[5])]} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
            img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            img_median = cv2.medianBlur(img_threshold, 5)

            # Dilater l'image pour rendre les lignes blanches plus épaisses
            kernel = np.ones((3, 3), np.uint8)
            img_dilate = cv2.dilate(img_median, kernel, iterations=1)

            # Détection des contours
            contours, _ = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 2)

            # Vérifier les espaces de parking
            check_parking_space(img_contours, img_dilate, contours)

            # Afficher l'image traitée
            cv2.imshow("Image", img_contours)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parking lot video or images.")
    parser.add_argument("--video", type=str, help="Path to the video file.")
    parser.add_argument("--images", type=str, nargs='*', help="Paths to the image files.")
    parser.add_argument("--positions", type=str, default="CarParkPos", help="Path to the parking positions file.")

    args = parser.parse_args()

    process_parking(video_path=args.video, image_paths=args.images, positions_path=args.positions)

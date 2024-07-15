import argparse
import pickle
from skimage.transform import resize
import numpy as np
import cv2

# Chemins par défaut
default_mask_path = "./data/mask_1920_1080.png"
default_model_path = './model/yuko_svc_best_est.p'

# Couleurs pour affichage
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
line_wt = 2

# Chargement du modèle
MODEL = pickle.load(open(default_model_path, "rb"))

def calc_diff(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2))

def get_parking_spots_boxes(cc):
    (total_con_comp, comp_label_ids, values, centroid) = cc
    slots = []
    coef = 1
    for i in range(1, total_con_comp):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

def get_coor(xywh):
    x1, y1, w, h = xywh
    top_left = (x1, y1)
    bottom_right = (x1 + w, y1 + h)
    return x1, y1, w, h, top_left, bottom_right

def is_empty(spot_bgr):
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = MODEL.predict(flat_data)
    return y_output == 0

def display_text(frame, spots_status):
    text = f"Available Lots: {str(sum(spots_status))} / {str(len(spots_status))}"
    white = (255, 255, 255)
    cv2.putText(frame, text, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)

def process_video(video_path, mask_path):
    mask = cv2.imread(mask_path, 0)
    cap = cv2.VideoCapture(video_path)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    parking_spots_xywh = get_parking_spots_boxes(connected_components)
    spots_status = [None for _ in parking_spots_xywh]
    prev_frame = None
    frame_num = 0
    ret = True
    frames_per_clf = 30

    while ret:
        ret, frame = cap.read()
        if frame_num % frames_per_clf == 0:
            if prev_frame is not None:
                diffs = [calc_diff(frame[y:y+h, x:x+w], prev_frame[y:y+h, x:x+w]) for (x, y, w, h) in parking_spots_xywh]
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.2]
            else:
                arr_ = range(len(parking_spots_xywh))
            for index in arr_:
                x, y, w, h = parking_spots_xywh[index]
                cropped_spot = frame[y:y+h, x:x+w, :]
                spots_status[index] = is_empty(cropped_spot)
        if frame_num % frames_per_clf == 0:
            prev_frame = frame.copy()
        for index, (x, y, w, h) in enumerate(parking_spots_xywh):
            color = green if spots_status[index] else red
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_wt)
        display_text(frame, spots_status)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_num += 1
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path, mask_path):
    mask = cv2.imread(mask_path, 0)
    frame = cv2.imread(image_path)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    parking_spots_xywh = get_parking_spots_boxes(connected_components)
    spots_status = [is_empty(frame[y:y+h, x:x+w, :]) for (x, y, w, h) in parking_spots_xywh]
    for index, (x, y, w, h) in enumerate(parking_spots_xywh):
        color = green if spots_status[index] else red
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_wt)
    display_text(frame, spots_status)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Parking Space Detection using Video or Image")
    parser.add_argument('--video', type=str, help="Path to the video file")
    parser.add_argument('--image', type=str, help="Path to the image file")
    parser.add_argument('--mask', type=str, default=default_mask_path, help="Path to the mask image")
    parser.add_argument('--model', type=str, default=default_model_path, help="Path to the trained model")
    args = parser.parse_args()

    if args.video:
        process_video(args.video, args.mask)
    elif args.image:
        process_image(args.image, args.mask)
    else:
        print("Please provide either a video or an image file path.")

if __name__ == "__main__":
    main()

"Ропізнаванння облич в реальному часі з відео-камери"

import cv2
import numpy as np
import time
import os
from pathlib import Path
import face_recognition
import traceback
from ultralytics import YOLO


class GenderClassifier:
    def __init__(self, photo_folder="models"):
        self.genders = ["Male", "Female"]
        if not os.path.exists(photo_folder):
            print(f"Папка не існує {photo_folder}")
        if not os.path.exists(os.path.join(f"{photo_folder}/gender_deploy.prototxt")):
            print(f"Файл {photo_folder}/gender_deploy.prototxt не знайдено!")
        if not os.path.exists(os.path.join(f"{photo_folder}/gender_net.caffemodel")):
            print(f"Файл {photo_folder}/gender_net.caffemodel не знайдено!")
        self.gender_net = cv2.dnn.readNetFromCaffe( f'{photo_folder}/gender_deploy.prototxt', f'{photo_folder}/gender_net.caffemodel')

    def predict(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426337, 87.768914, 114.895847), swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()[0]
        index = np.argmax(gender_preds)
        gender = self.genders[index]
        confidence = gender_preds[index] * 100
        return gender, confidence

class FaceRecognitionSystem:
    def __init__(self, photo_folder="models"):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        # детекція статі
        self.gender_classifier = GenderClassifier()
        self.face_genders = []
        # детекція об'єктів (банан)
        if not os.path.exists(photo_folder):
            print(f"Папка не існує {photo_folder}")
        if not os.path.exists(os.path.join(f"{photo_folder}/yolov8n.pt")):
            print(f"Файл {photo_folder}/yolov8n.pt не знайдено!")
        self.object_detector = YOLO(f"{photo_folder}/yolov8n.pt")

    def load_known_faces(self, photo_folder="photos"):
        print("Завантаження фотографій для навчання...")
        if not os.path.exists(photo_folder):
            print(f"Папка не існує {photo_folder}")

        person_folders = [f for f in Path(photo_folder).iterdir() if f.is_dir()]

        if not person_folders:
            print(f"Не знайдено підпапок з працівниками в діректорії -  {photo_folder}")
            return False

        total_photos = 0

        for person_folder in person_folders:
            person_name = person_folder.name
            photo_files = (list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg"))
                           + list(person_folder.glob("*.png")))

            if not photo_files:
                print(f"Немає фото {person_name}. ПРОПУШЕНО!")
                continue

            print(f"\n Обробка: {person_name}")
            for photo_path in photo_files:
                try:
                    # Завантажуємо зображення
                    image = face_recognition.load_image_file(str(photo_path))

                    # Знаходимо обличчя
                    face_encodings = face_recognition.face_encodings(image)

                    if len(face_encodings) == 0:
                        print(f"    - {photo_path.name}: Обличчя не знайдено!")
                        continue
                    elif len(face_encodings) > 1:
                        print(f"    - {photo_path.name}: Знайдено кілька облич")

                    # Додаємо в базу
                    self.known_faces_encodings.append(face_encodings[0])
                    self.known_faces_names.append(person_name)
                    total_photos += 1
                    print(f" + {photo_path.name}")

                except Exception as err:
                    print("Сталась помилка! --- ", err)

        if total_photos == 0:
            print("\n Не вдалось завантажити жодного обличчя")
            return False

        print(f" Успішно завантажено {total_photos} фото облич")
        return True


    def progress_frame(self, frame, scale_factor= 0.25):
        self.face_genders = []  # очищуємо список статей
        # Зменшуємо кадр для швидшої обробки
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) # перетворюємо в RGB

        # Знаходимо обличчя на кадрі
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []

        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(
                self.known_faces_encodings,
                face_encoding,
                tolerance=0.6
            )
            name = "Unknown Person"
            confidence = 0

            face_distances = face_recognition.face_distance(
                self.known_faces_encodings,
                face_encoding
            )

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_faces_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100

            self.face_names.append((name, confidence))
            # Визначення статі
            (top, right, bottom, left) = self.face_locations[len(self.face_names) - 1]
            face_crop = rgb_small_frame[top:bottom, left:right]
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            gender, gender_conf = self.gender_classifier.predict(face_crop)
            self.face_genders.append((gender, gender_conf))

        self.face_locations = [
                (int(top/scale_factor), int(right/scale_factor), int(bottom/scale_factor), int(left/scale_factor))
                for (top, right, bottom, left) in self.face_locations
            ]

        self.detected_objects = []

        results = self.object_detector(frame)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            if label.lower() == "banana":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.detected_objects.append((x1, y1, x2, y2, label, conf))

    def draw_results(self, frame):
        for ((top, right, bottom, left), (name, confidence),
                (gender, gender_conf)
             ) in zip(
                self.face_locations, self.face_names, self.face_genders
            ):
            if name == "Unknown Person":
                color = (0, 0, 255) # red
            else:
                color = (0, 255, 0) # green

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)   # рамка
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED) # фон тексту

            # Імя та впевненість
            font = cv2.FONT_HERSHEY_DUPLEX
            if confidence > 0:
                text = f"{name} ({confidence:.1f}%) - {gender} ({gender_conf:.1f}%)"
            else:
                text = f"{name} - {gender} ({gender_conf:.1f}%)"

            cv2.putText(
                frame, text, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1
            )

        # Малюємо знайдені об'єкти
        for (x1, y1, x2, y2, label, conf) in self.detected_objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} ({conf * 100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def run_face_recognition(camera_id=0, photos_folder="photos"):
    system = FaceRecognitionSystem()
    cap = cv2.VideoCapture(camera_id)

    if not system.load_known_faces(photos_folder):
        print("Не вдалось завантажити обличчя!")
        return

    if not cap.isOpened():
        print("Не вдалось відкрити камеру!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("РОЗПІЗНАВАННЯ ЗАПУЩЕНО!")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    progress_this_frame = True # для перевірки кожного другого кадру
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Помилка читання кадру!")
            break

        if progress_this_frame:
            system.progress_frame(frame, scale_factor=0.25)

        progress_this_frame = not progress_this_frame

        # Малюємо результат
        system.draw_results(frame)

        # FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        cv2.rectangle(frame, (5, 5), (150, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Показуємо кадр
        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF

    cap.release()
    cv2.destroyAllWindows()
    print("Робота завершена!")

if __name__ == "__main__":
    try:
        run_face_recognition(camera_id=0, photos_folder="photos")
    except KeyboardInterrupt:
        print("Робота перервана користувачем!")
    except Exception as err:
        print(" Error:", err)
        traceback.print_exc()
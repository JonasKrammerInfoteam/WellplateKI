import cv2
import numpy as np
import os
import math

min_dist = 20
param1 = 50
param2 = 25
min_radius = 15
max_radius = 25

def detect(image):
    wells = find_wells(image)
    if wells is None:
        return image
    else:
        return find_fillings(image, wells)

def find_wells(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wells = cv2.HoughCircles(grayscale_img, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_dist,
                            param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return wells

def find_fillings(image, circles):

    # Bild in den HSV-Farbraum konvertieren
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Schritt 5: Verstärke nur die Rottöne


    # Rote Bereiche im HSV-Bereich definieren
    #lower_red = np.array([0, 50, 150])
    lower_red = np.array([0, 5, 0])
    upper_red = np.array([10, 255, 255])
    # Maske für rote Bereiche erstellen
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    circles = np.round(circles[0, :]).astype("int")

    filled_wells = 0
    total_wells = circles.shape[0]
    for (x, y, r) in circles:
        # Extrahiere die Region des Kreises
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)  # Erstelle eine Maske für den Kreis
        
        # Maske auf das Bild anwenden, um den Kreisbereich zu extrahieren
        circle_region = cv2.bitwise_and(red_mask, red_mask, mask=mask[:, :, 0])

        # Berechne den Anteil roter Pixel im Kreisbereich
        red_area = np.sum(circle_region == 255)  # Zähle die weißen (roten) Pixel in der Region
        total_area = np.pi * r**2  # Gesamtfläche des Kreises

        # Schwellenwert für den Anteil roter Fläche festlegen (z.B. 20%)
        red_ratio = red_area / total_area

        # Prüfen, ob der Kreis eine rote Füllung hat
        if red_ratio > 0.01:  # 20% der Fläche sind rot (Schwellenwert anpassbar)
            filled_wells += 1
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Markiere Kreise mit roter Füllung
            cv2.putText(image, f"Red: {int(red_ratio*100)}%", (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"Total: {total_wells}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Filled: {filled_wells}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

if __name__ == '__main__':
    live = True
    test = True
    if live == False:
        if test:
            current_test_image = cv2.imread("camera_image.png")
            result_image = detect(current_test_image)
            cv2.imshow("Detected Wells", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            for filename in os.listdir("images"):
                image_filename = os.path.join("images", filename)
                # Überprüfen, ob es sich um eine Bilddatei handelt
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    current_test_image = cv2.imread(image_filename)
                    result_image = detect(current_test_image)
                    cv2.imshow("Detected Wells", result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        # Starte den Kamera-Stream
        # Video speichern: Dateiname, Codec, FPS, Frame-Größe
        video_filename = "wellplate_detection.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (alternativ: 'MP4V' für .mp4)
        fps = 20
        frame_size = (640, 480)  # Muss mit Kamera-Frame-Größe übereinstimmen!

        # VideoWriter initialisieren
        out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
        cap = cv2.VideoCapture(1)  # 0 für Standard-Webcam, ggf. andere Zahl für externe Kamera

        saved_image = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result_image = detect(frame)
            
            # Video speichern
            out.write(result_image)
            # Zeige das Live-Bild
            cv2.imshow("Live Well Detection", result_image)

            # Beende das Programm, wenn 'q' gedrückt wird
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.imwrite("camera_image.png", result_image)
                break

        # Kamera-Stream beenden
        cap.release()
        out.release()
        cv2.destroyAllWindows()
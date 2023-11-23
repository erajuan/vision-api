import cv2
import time
import math
import numpy as np
import mediapipe as mp
import onnxruntime


def Distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist


def mouth_aspect_ratio(p1, p2, p3, p4, p5, p6, p7, p8):
    d_A = Distance(p2, p8)
    d_B = Distance(p3, p7)
    d_C = Distance(p4, p6)
    d_D = Distance(p5, p1)
    return (d_A + d_B + d_C) / (3 * d_D)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

left_x, left_y, left_w, left_h = 0, 0, 0, 0
right_x, right_y, right_w, right_h = 0, 0, 0, 0


clases = {0: "Undetected", 1: "DETECTED"}

providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
session = onnxruntime.InferenceSession("model_best_ddaig.onnx", providers=providers)
session.get_inputs()[0].shape
session.get_inputs()[0].type
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def is_the_person_asleep(filename: str):
    person = {}
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.75, min_tracking_confidence=0.5
    ) as face_mesh:
        frame = cv2.imread(filename=filename)

        frame1 = frame.copy()
        roi = frame1.copy()
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                puntos = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame1.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    puntos.append([x, y])

                if puntos:
                    ojoder1 = puntos[336]
                    ojoder2 = puntos[293]
                    ojoder3 = puntos[346]

                    ojoizq1 = puntos[63]
                    ojoizq2 = puntos[107]
                    ojoizq3 = puntos[117]

                    dedW = Distance(ojoder1, ojoder2)
                    dedH = Distance(ojoder2, ojoder3)
                    deiW = Distance(ojoizq1, ojoizq2)
                    deiH = Distance(ojoizq1, ojoizq3)

                    xed = puntos[336][0]
                    yed = puntos[336][1]
                    xei = puntos[63][0]
                    yei = puntos[63][1]

                    right_x, right_y, right_w, right_h = xed, yed, int(dedW), int(dedH)
                    left_x, left_y, left_w, left_h = xei, yei, int(deiW), int(deiH)

                    if left_x > right_x:
                        start_x, end_x = right_x, (left_x + left_w)
                    else:
                        start_x, end_x = left_x, (right_x + right_w)

                    if left_y > right_y:
                        start_y, end_y = right_y, (left_y + left_h)
                    else:
                        start_y, end_y = left_y, (right_y + right_h)

                    if (end_x - start_x) > 50 and (end_y - start_y) < 200:
                        start_x, start_y, end_x, end_y = (
                            start_x - 10,
                            start_y - 10,
                            end_x + 10,
                            end_y + 10,
                        )
                        # cv2.rectangle(frame1, (start_x, start_y), (end_x, end_y), (246, 255, 0), 2)
                        face_roi = roi[start_y:end_y, start_x:end_x]
                        face_roi = cv2.resize(face_roi, (64, 64))
                        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        img_rsp = face_roi_gray.reshape(1, 64, 64, 1)
                        img_rsp = (img_rsp / 255.0).astype(np.float32)

                        prediction = session.run([output_name], {input_name: img_rsp})
                        predicted_class_onnx = np.argmax(prediction, axis=None)
                        prediction_label_onnx = clases[predicted_class_onnx]
                        nmax = np.max(prediction)
                        prob = round(nmax, 3) * 100
                        person["class"] = prediction_label_onnx
                        person["probability"] = prob
    return person
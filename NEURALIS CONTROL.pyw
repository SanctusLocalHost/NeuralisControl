import os
import sys
import warnings

# --- SUPRESSÃO DE AVISOS (MÉTODO ROBUSTO) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')
import mediapipe as mp
sys.stderr.close()
sys.stderr = stderr_original
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
import time
import platform
import math
import pyautogui
import threading
import random

# --- Carregamento Robusto de Sons ---
try:
    import pygame
    pygame.mixer.init()
    # Ajuste os caminhos conforme necessário
    sound_activate = pygame.mixer.Sound(r"G:\Meu Drive\CONTROLLER\DATA CENTER\SOM\ATIVED_NEURAL_CONTROL.ogg")
    sound_deactivate = pygame.mixer.Sound(r"G:\Meu Drive\CONTROLLER\DATA CENTER\SOM\UNATIVED_NEURAL_CONTROL.ogg")
    sound_click = pygame.mixer.Sound(r"G:\Meu Drive\CONTROLLER\DATA CENTER\SOM\CLICK_NEURAL_CONTROL.ogg")
    sound_enabled = True
except (ImportError, FileNotFoundError, pygame.error) as e:
    print("\nAVISO: Pygame ou um dos arquivos de som (.ogg) nao foram encontrados.")
    print("O feedback sonoro esta desativado. O programa continuara funcionando normalmente.")
    sound_enabled = False

# --- Tenta importar bibliotecas específicas para o sistema ---
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    import ctypes

# =====================================================================================
# --- PARÂMETROS DE CONFIGURAÇÃO ---
# =====================================================================================
W_CAM, H_CAM = 1280, 720
SENSITIVITY = 7.5
SMOOTHING_FACTOR = 0.15
ACTION_DELAY_SECONDS = 0.5
DRAG_COOLDOWN_SECONDS = 2.0
TOGGLE_HOLD_SECONDS = 0.5
# =====================================================================================

# --- Variáveis Globais de Estado ---
program_is_running = True
camera_window_visible = False
lock = threading.Lock()
is_gesture_control_active = False

def user_input_handler():
    global program_is_running, camera_window_visible
    while program_is_running:
        try:
            command = input().lower()
            with lock:
                if command == 'c':
                    camera_window_visible = not camera_window_visible
                    if camera_window_visible: print("--> Comando: Mostrar Visualizador.")
                    else: print("--> Comando: Esconder Visualizador.")
                elif command == 'q':
                    print("--> Comando: Encerrando Interface...")
                    program_is_running = False
        except (EOFError, KeyboardInterrupt):
            with lock: program_is_running = False
            break

def move_mouse_fast(x, y):
    if IS_WINDOWS: ctypes.windll.user32.SetCursorPos(int(x), int(y))
    else: pyautogui.moveTo(x, y)

def get_finger_state(hand_landmarks, handedness_label):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    is_right_hand = handedness_label == 'Right'
    
    if is_right_hand:
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x: fingers.append(1)
        else: fingers.append(0)
    else:
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x: fingers.append(1)
        else: fingers.append(0)
    
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y: fingers.append(1)
        else: fingers.append(0)
    return fingers

def display_boot_sequence():
    """Exibe a interface de inicialização estilizada no terminal."""
    print("\nBooting NEURALIS INTERFACE...")
    time.sleep(0.5)
    print("[*] Loading Neural Link Subsystems...")
    time.sleep(0.7)
    print("[+] Gesture Recognition Matrix... OK")
    print("[+] Face Detection Module... OK")
    print("[+] Media Control Protocol... OK")
    print("[+] Audio Feedback Module... " + ("OK" if sound_enabled else "DISABLED"))
    time.sleep(0.5)
    print("\n=======================================================")
    print("    NEURALIS - CONTROL [ONLINE]")
    print("-------------------------------------------------------")
    print("INFO: -> System starting in [INACTIVE] state.")
    print("       -> Perform activation gesture (2 Hands Open) to engage.")
    print("\n[GESTURE GUIDE]")
    print("  Right Hand Open : Play/Pause Media (Silent)")
    print("  Index+Middle    : Move Cursor")
    print("  Index+Middle+Thumb : Drag Mode")
    print("  Rock Sign       : Click")
    print("\n[TERMINAL COMMANDS]")
    print("  'c' + Enter : Toggle Visualizer Window")
    print("  'q' + Enter : Terminate Program")
    print("=======================================================\n")

# --- INICIALIZAÇÃO ---
cap = cv2.VideoCapture(0)
cap.set(3, W_CAM)
cap.set(4, H_CAM)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Variáveis de estado
prev_hand_x, prev_hand_y, smoothed_delta_x, smoothed_delta_y = 0, 0, 0, 0
is_moving_mode, is_mouse_down, action_locked = False, False, False
p_time = 0
current_gesture, gesture_start_time = "NEUTRAL", 0
drag_cooldown_end_time = 0
flash_effect_end_time = 0
prev_scroll_y = 0

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False
input_thread = threading.Thread(target=user_input_handler, daemon=True)
input_thread.start()

display_boot_sequence()

try:
    while program_is_running:
        success, img = cap.read()
        if not success: continue
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results_hands = hands.process(img_rgb)
        results_face = face_detection.process(img_rgb)

        raw_detected_gesture = "NEUTRAL"
        all_hands_data = []
        cx1, cy1, cx2, cy2 = 0, 0, 0, 0

        # --- PROCESSAMENTO DAS MÃOS ---
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                handedness_label = handedness_info.classification[0].label
                fingers = get_finger_state(hand_landmarks, handedness_label)
                all_hands_data.append({'landmarks': hand_landmarks, 'fingers': fingers, 'label': handedness_label})
                if camera_window_visible:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            num_hands = len(all_hands_data)
            
            # --- LÓGICA DE 2 MÃOS ---
            if num_hands == 2:
                if all_hands_data[0]['label'] != all_hands_data[1]['label']:
                    fingers1 = all_hands_data[0]['fingers']
                    fingers2 = all_hands_data[1]['fingers']
                    if fingers1 == [1, 1, 1, 1, 1] and fingers2 == [1, 1, 1, 1, 1]:
                        raw_detected_gesture = "TOGGLE_INTENT"
                    elif is_gesture_control_active:
                        if fingers1 == [0, 1, 1, 0, 0] and fingers2 == [0, 1, 1, 0, 0]: raw_detected_gesture = "ZOOM"
                        elif fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]: raw_detected_gesture = "ARCANEDIAL_MODE"
            
            # --- LÓGICA DE 1 MÃO ---
            if is_gesture_control_active and num_hands == 1:
                fingers = all_hands_data[0]['fingers']
                hand_label = all_hands_data[0]['label']

                # Prioridade de gestos
                if fingers == [0, 1, 1, 0, 0]: raw_detected_gesture = "MOVE"
                elif fingers == [1, 1, 1, 0, 0]: raw_detected_gesture = "DRAG"
                elif fingers == [0, 1, 1, 0, 1]: raw_detected_gesture = "CLICK_INTENT"
                # Mão Aberta e Direita = Media Control
                elif fingers == [1, 1, 1, 1, 1] and hand_label == "Right":
                    raw_detected_gesture = "MEDIA_CONTROL"

        # --- GESTURE DEBOUNCING ---
        if raw_detected_gesture != current_gesture:
            if current_gesture == "DRAG":
                drag_cooldown_end_time = time.time() + DRAG_COOLDOWN_SECONDS
                if is_mouse_down: pyautogui.mouseUp(); is_mouse_down = False
            
            if raw_detected_gesture == "DRAG" and time.time() < drag_cooldown_end_time:
                current_gesture = "DRAG_COOLDOWN"
            else:
                current_gesture = raw_detected_gesture
            
            gesture_start_time = time.time()
            action_locked = False
        
        # --- EXECUÇÃO DAS AÇÕES ---
        if is_gesture_control_active:
            is_movement_gesture = (current_gesture == "MOVE") or (current_gesture == "DRAG")
            
            # MOVIMENTO
            if is_movement_gesture and all_hands_data:
                lm = all_hands_data[0]['landmarks']
                ix, iy = lm.landmark[8].x * W_CAM, lm.landmark[8].y * H_CAM
                mx, my = lm.landmark[12].x * W_CAM, lm.landmark[12].y * H_CAM
                current_hand_x, current_hand_y = (ix + mx) / 2, (iy + my) / 2
                if not is_moving_mode:
                    is_moving_mode = True
                    prev_hand_x, prev_hand_y = current_hand_x, current_hand_y
                else:
                    delta_x = current_hand_x - prev_hand_x
                    delta_y = current_hand_y - prev_hand_y
                    smoothed_delta_x = (smoothed_delta_x * (1 - SMOOTHING_FACTOR)) + (delta_x * SMOOTHING_FACTOR)
                    smoothed_delta_y = (smoothed_delta_y * (1 - SMOOTHING_FACTOR)) + (delta_y * SMOOTHING_FACTOR)
                    mouse_x, mouse_y = pyautogui.position()
                    new_mouse_x = mouse_x + smoothed_delta_x * SENSITIVITY
                    new_mouse_y = mouse_y + smoothed_delta_y * SENSITIVITY
                    move_mouse_fast(new_mouse_x, new_mouse_y)
                    prev_hand_x, prev_hand_y = current_hand_x, current_hand_y
            else:
                is_moving_mode = False

            # CLIQUE
            if current_gesture == "CLICK_INTENT":
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time >= ACTION_DELAY_SECONDS and not action_locked:
                    pyautogui.click()
                    if sound_enabled: sound_click.play()
                    action_locked = True
                    flash_effect_end_time = time.time() + 0.15
            
            # DRAG
            elif current_gesture == "DRAG":
                if not is_mouse_down:
                    pyautogui.mouseDown(); is_mouse_down = True
            
            # MEDIA CONTROL (PLAY/PAUSE) - SEM SOM
            elif current_gesture == "MEDIA_CONTROL":
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time >= ACTION_DELAY_SECONDS and not action_locked:
                    pyautogui.press('playpause')
                    print("--> Mídia: Play/Pause acionado.")
                    # if sound_enabled: sound_click.play() # REMOVIDO PARA EVITAR SOM
                    action_locked = True
                    flash_effect_end_time = time.time() + 0.15

            # SCROLL
            elif current_gesture == "ARCANEDIAL_MODE":
                lm1, lm2 = all_hands_data[0]['landmarks'], all_hands_data[1]['landmarks']
                cy1, cy2 = lm1.landmark[8].y * H_CAM, lm2.landmark[8].y * H_CAM
                active_hand_y = min(cy1, cy2)
                if prev_scroll_y == 0: prev_scroll_y = active_hand_y
                delta_y = active_hand_y - prev_scroll_y
                if abs(delta_y) > 2: pyautogui.scroll(int(-delta_y * 1.5))
                prev_scroll_y = active_hand_y
            
            if current_gesture != "ARCANEDIAL_MODE": prev_scroll_y = 0

            # ZOOM
            if current_gesture == "ZOOM":
                lm1, lm2 = all_hands_data[0]['landmarks'], all_hands_data[1]['landmarks']
                ix1, iy1, mx1, my1 = lm1.landmark[8].x, lm1.landmark[8].y, lm1.landmark[12].x, lm1.landmark[12].y
                ix2, iy2, mx2, my2 = lm2.landmark[8].x, lm2.landmark[8].y, lm2.landmark[12].x, lm2.landmark[12].y
                cx1, cy1 = (ix1 + mx1) / 2 * W_CAM, (iy1 + my1) / 2 * H_CAM
                cx2, cy2 = (ix2 + mx2) / 2 * W_CAM, (iy2 + my2) / 2 * H_CAM
                dist = math.hypot(cx2 - cx1, cy2 - cy1)
                if initial_zoom_dist == 0: initial_zoom_dist = dist
                zoom_amount = (dist - initial_zoom_dist) / 200.0
                pyautogui.scroll(int(zoom_amount * 10))
            else:
                initial_zoom_dist = 0
        
        # --- TOGGLE SYSTEM ---
        if current_gesture == "TOGGLE_INTENT":
            elapsed_time = time.time() - gesture_start_time
            if elapsed_time >= TOGGLE_HOLD_SECONDS and not action_locked:
                is_gesture_control_active = not is_gesture_control_active
                if sound_enabled:
                    if is_gesture_control_active: sound_activate.play()
                    else: sound_deactivate.play()
                action_locked = True

        # --- RENDERIZAÇÃO VISUAL ---
        if camera_window_visible:
            if is_gesture_control_active:
                cv2.putText(img, "SISTEMA ATIVO", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                
                if current_gesture == "MOVE" and all_hands_data:
                    lm = all_hands_data[0]['landmarks']
                    ix, iy = lm.landmark[8].x * W_CAM, lm.landmark[8].y * H_CAM
                    pulse_radius = 15 + int(10 * abs(math.sin(time.time() * 5)))
                    cv2.circle(img, (int(ix), int(iy)), pulse_radius, (255, 0, 255), 2)
                    cv2.putText(img, "FOCO ARCANO", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                
                elif current_gesture == "CLICK_INTENT":
                    elapsed_time = time.time() - gesture_start_time
                    if not action_locked and all_hands_data:
                        charge_radius = int(80 * (1 - (elapsed_time / ACTION_DELAY_SECONDS)))
                        lm = all_hands_data[0]['landmarks']
                        wx, wy = lm.landmark[0].x * W_CAM, lm.landmark[0].y * H_CAM
                        cv2.circle(img, (int(wx), int(wy)), charge_radius, (0, 255, 255), 3)
                        cv2.putText(img, "CANALIZANDO...", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                
                # VISUAL MEDIA CONTROL
                elif current_gesture == "MEDIA_CONTROL" and all_hands_data:
                    elapsed_time = time.time() - gesture_start_time
                    if not action_locked:
                        lm = all_hands_data[0]['landmarks']
                        wx, wy = lm.landmark[9].x * W_CAM, lm.landmark[9].y * H_CAM
                        progress = min(elapsed_time / ACTION_DELAY_SECONDS, 1.0)
                        cv2.ellipse(img, (int(wx), int(wy)), (50, 50), 0, 0, 360 * progress, (0, 255, 0), 5)
                        cv2.putText(img, "COMANDO DE MIDIA", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                
                if time.time() < flash_effect_end_time:
                    cv2.circle(img, (int(W_CAM/2), int(H_CAM/2)), int(W_CAM), (255, 255, 255), -1)
                    if current_gesture == "MEDIA_CONTROL":
                        cv2.putText(img, "PLAY / PAUSE", (int(W_CAM/2)-150, int(H_CAM/2)), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)
                    else:
                        cv2.putText(img, "IMPACTO!", (int(W_CAM/2)-100, int(H_CAM/2)), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)
                
                elif current_gesture == "DRAG" and all_hands_data:
                    lm = all_hands_data[0]['landmarks']
                    tx, ty = lm.landmark[4].x * W_CAM, lm.landmark[4].y * H_CAM
                    mx, my = lm.landmark[12].x * W_CAM, lm.landmark[12].y * H_CAM
                    anchor_x = (tx + mx) / 2
                    cv2.line(img, (int(anchor_x), int(my)), (int(anchor_x), 0), (255, 0, 255), 8)
                    cv2.putText(img, "ELO ESPECTRAL", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

                elif current_gesture in ["ZOOM", "ARCANEDIAL_MODE"] and len(all_hands_data) == 2:
                    if current_gesture == "ZOOM":
                        lm1, lm2 = all_hands_data[0]['landmarks'], all_hands_data[1]['landmarks']
                        ix1, iy1 = (lm1.landmark[8].x + lm1.landmark[12].x) / 2 * W_CAM, (lm1.landmark[8].y + lm1.landmark[12].y) / 2 * H_CAM
                        ix2, iy2 = (lm2.landmark[8].x + lm2.landmark[12].x) / 2 * W_CAM, (lm2.landmark[8].y + lm2.landmark[12].y) / 2 * H_CAM
                        text, color = "FENDA ESPACIAL", (255, 0, 255)
                    else: 
                        lm1, lm2 = all_hands_data[0]['landmarks'], all_hands_data[1]['landmarks']
                        ix1, iy1 = lm1.landmark[8].x * W_CAM, lm1.landmark[8].y * H_CAM
                        ix2, iy2 = lm2.landmark[8].x * W_CAM, lm2.landmark[8].y * H_CAM
                        text, color = "DIAL ARCANO", (255, 255, 0)
                    aura_color = (255, 255, 0)
                    cv2.circle(img, (int(ix1), int(iy1)), 20, aura_color, 2)
                    cv2.circle(img, (int(ix2), int(iy2)), 20, aura_color, 2)
                    cv2.line(img, (int(ix1), int(iy1)), (int(ix2), int(iy2)), color, 4)
                    for _ in range(5):
                        if ix1 != ix2:
                            spark_x = random.randint(int(min(ix1, ix2)), int(max(ix1, ix2)))
                            spark_y = int(np.interp(spark_x, [ix1, ix2], [iy1, iy2]))
                            cv2.circle(img, (spark_x, spark_y), 3, (255, 255, 255), -1)
                    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
            else:
                cv2.putText(img, "SISTEMA INATIVO", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            
            if current_gesture == "TOGGLE_INTENT" and not action_locked:
                if is_gesture_control_active:
                    text, color = "Segure para DESATIVAR...", (0, 100, 255)
                else:
                    text, color = "Segure para ATIVAR...", (0, 255, 255)
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            cv2.putText(img, f'FPS: {int(fps)}', (W_CAM - 150, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            if results_face.detections:
                for detection in results_face.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Humano", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("Neuralis Interface", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                program_is_running = False
        else:
            cv2.destroyAllWindows()
            p_time = time.time()
            time.sleep(0.01)

finally:
    print("Recursos liberados. Programa encerrado.")
    program_is_running = False
    cap.release()
    cv2.destroyAllWindows()

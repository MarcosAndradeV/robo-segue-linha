import cv2
import mediapipe as mp


def detect_hands():
    # Inicialização da captura de vídeo
    cap = cv2.VideoCapture(0)

    # Inicialização do mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2, min_detection_confidence=0.5)

    while True:
        # Captura de um frame
        ret, frame = cap.read()

        # Conversão para grayscale para o mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecção das mãos
        results = hands.process(image)

        # Desenhar as mãos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Exibindo o frame resultante
        cv2.imshow('Detecção de Mãos', frame)

        # Saindo do loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberando a captura de vídeo e fechando todas as janelas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_hands()

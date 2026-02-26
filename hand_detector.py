import cv2
import mediapipe as mp
import numpy as np


def get_finger_states(hand_landmarks, hand_label):
    """
    返回一个长度为 5 的列表，分别表示 [拇指, 食指, 中指, 无名指, 小指] 是否伸出。
    1 = 伸出, 0 = 弯曲。

    拇指判断通过食指根部和小指根部的 X 位置关系来动态确定伸出方向，
    这样无论手心还是手背朝向摄像头都能正确判断。
    """
    fingers = []
    lm = hand_landmarks.landmark

    # ---- 拇指 ----
    # 食指 MCP(5) 和小指 MCP(17) 形成手掌的横向排列
    # 拇指总是在食指 MCP 的同侧，并且伸出时会朝着远离手掌中心的方向延伸
    idx_mcp_x = lm[5].x
    pinky_mcp_x = lm[17].x

    if idx_mcp_x > pinky_mcp_x:
        # 食指在右侧 → 拇指也在右侧 → 伸出时指尖(4) X 大于 MCP(2) X
        fingers.append(1 if lm[4].x > lm[2].x else 0)
    else:
        # 食指在左侧 → 拇指也在左侧 → 伸出时指尖(4) X 小于 MCP(2) X
        fingers.append(1 if lm[4].x < lm[2].x else 0)

    # ---- 食指、中指、无名指、小指 ----
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)

    return fingers


def is_middle_finger_only(finger_states):
    """
    判断是否只有中指竖起（竖中指手势）。
    finger_states 顺序: [拇指, 食指, 中指, 无名指, 小指]

    忽略拇指状态（手背朝向时拇指检测不稳定），
    只要求：食指弯、中指伸、无名指弯、小指弯。
    """
    return (finger_states[1] == 0 and finger_states[2] == 1
            and finger_states[3] == 0 and finger_states[4] == 0)


def blur_hand_region(frame, hand_landmarks, h, w, padding=40):
    """
    根据手部关键点的包围框，对该区域进行高斯模糊。
    padding: 在包围框四周额外扩展的像素数，确保整只手被完全覆盖。
    """
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x_min = max(0, min(x_coords) - padding)
    y_min = max(0, min(y_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_max = min(h, max(y_coords) + padding)

    # 用强力高斯模糊完全遮盖手部区域
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y_min:y_max, x_min:x_max] = blurred

    return frame


def main():
    # ---------- 初始化 MediaPipe Hands ----------
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,     # 视频流模式，启用追踪以提高效率
        max_num_hands=2,             # 同时识别最多 2 只手
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # ---------- 打开摄像头 ----------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备是否已连接。")
        return

    print("程序已启动，正在调用摄像头进行手部识别...")
    print("按 'q' 键退出。")

    while True:
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败，程序退出。")
            break

        # 水平翻转画面，让用户体验更像照镜子
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe 需要 RGB 输入
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            total_fingers = 0
            has_offensive = False

            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                # 获取左/右手标签
                label = results.multi_handedness[idx].classification[0].label

                # 获取每根手指的伸出状态
                finger_states = get_finger_states(hand_lm, label)
                finger_count = sum(finger_states)

                # ===== 竖中指检测 =====
                if is_middle_finger_only(finger_states):
                    has_offensive = True
                    # 模糊掉该手部区域
                    frame = blur_hand_region(frame, hand_lm, h, w)
                    # 不绘制骨架，只在手腕下方显示警告
                    wrist_x = int(hand_lm.landmark[0].x * w)
                    wrist_y = int(hand_lm.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        "BLOCKED",
                        (wrist_x - 50, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
                else:
                    # 正常手势 —— 绘制骨架和手指数
                    total_fingers += finger_count
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    wrist_x = int(hand_lm.landmark[0].x * w)
                    wrist_y = int(hand_lm.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        f"{label} Hand: {finger_count}",
                        (wrist_x - 40, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            # 顶部状态栏
            if has_offensive:
                cv2.putText(
                    frame,
                    "WARNING!!!!",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    f"Total Fingers: {total_fingers}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    3,
                )
        else:
            cv2.putText(
                frame,
                "No hands detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        cv2.imshow("Hand & Finger Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---------- 释放资源 ----------
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
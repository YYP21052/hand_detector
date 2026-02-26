# -*- coding: utf-8 -*-
"""
hand_detector.py - 基于MediaPipe的手势识别与不雅手势过滤

功能说明：
    使用Google MediaPipe Hands模型实时检测手部21个关键点，
    根据关键点坐标判断每根手指的伸出/弯曲状态，从而计算手指数量。
    同时内置了竖中指手势的识别功能，一旦检测到该手势，
    会自动对手部区域进行高斯模糊遮挡，并显示警告信息。

使用方式：
    直接运行本文件即可启动摄像头窗口，按 'q' 键退出。

依赖库：
    opencv-python, mediapipe, numpy
"""

import cv2
import mediapipe as mp
import numpy as np


def get_finger_states(hand_landmarks, hand_label):
    """
    判断5根手指的伸出状态。

    参数：
        hand_landmarks: MediaPipe检测到的手部关键点对象
        hand_label: 左手("Left")或右手("Right")的标签

    返回：
        长度为5的列表 [拇指, 食指, 中指, 无名指, 小指]
        1表示伸出，0表示弯曲
    """
    fingers = []
    lm = hand_landmarks.landmark

    # ---- 拇指判断 ----
    # 拇指比较特殊，不能简单用y坐标判断，需要看x方向
    # 通过比较食指根部(5号点)和小指根部(17号点)的x坐标
    # 来判断手掌的朝向，从而确定拇指伸出的方向
    idx_mcp_x = lm[5].x    # 食指根部的x坐标
    pinky_mcp_x = lm[17].x  # 小指根部的x坐标

    if idx_mcp_x > pinky_mcp_x:
        # 食指在小指右边 → 拇指伸出时指尖(4号)应该在第一关节(2号)的右边
        fingers.append(1 if lm[4].x > lm[2].x else 0)
    else:
        # 食指在小指左边 → 拇指伸出时指尖应该在第一关节的左边
        fingers.append(1 if lm[4].x < lm[2].x else 0)

    # ---- 其余四根手指判断 ----
    # 食指(8)、中指(12)、无名指(16)、小指(20)的指尖landmark编号
    # 判断逻辑：指尖的y坐标 < 中间关节(tip_id - 2)的y坐标 → 手指伸直
    # 注意：MediaPipe坐标系中y轴向下为正，所以y值越小代表越靠上
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)

    return fingers


def is_middle_finger_only(finger_states):
    """
    判断是否为竖中指手势。

    判断条件（忽略拇指，因为手背朝向摄像头时拇指的检测结果不稳定）：
        - 食指弯曲、中指伸出、无名指弯曲、小指弯曲

    参数：
        finger_states: get_finger_states()的返回值，[拇指, 食指, 中指, 无名指, 小指]

    返回：
        True表示检测到竖中指，False表示不是
    """
    return (finger_states[1] == 0 and finger_states[2] == 1
            and finger_states[3] == 0 and finger_states[4] == 0)


def blur_hand_region(frame, hand_landmarks, h, w, padding=40):
    """
    对手部所在区域做高斯模糊遮挡。

    根据手部所有关键点的坐标算出一个矩形包围框，
    向四周各扩展padding个像素后，对该区域施加强力高斯模糊。

    参数：
        frame: 当前视频帧
        hand_landmarks: 手部关键点
        h, w: 帧的高度和宽度
        padding: 包围框向外扩展的像素数

    返回：
        处理后的帧
    """
    # 把归一化坐标(0~1)转换成像素坐标
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    # 计算包围框，同时做边界裁剪防止越界
    x_min = max(0, min(x_coords) - padding)
    y_min = max(0, min(y_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_max = min(h, max(y_coords) + padding)

    # 对包围框内的区域做高斯模糊（核大小99x99，模糊力度很强）
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y_min:y_max, x_min:x_max] = blurred

    return frame


def main():
    # ==================== 初始化MediaPipe手部检测模型 ====================
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils       # 用来画手部骨架的工具
    mp_styles = mp.solutions.drawing_styles     # 骨架的默认绘制样式

    # 创建手部检测器实例
    hands = mp_hands.Hands(
        static_image_mode=False,     # False=视频流模式，会启用帧间追踪，效率更高
        max_num_hands=2,             # 最多同时检测2只手
        min_detection_confidence=0.7,  # 检测阈值，越高越严格
        min_tracking_confidence=0.5,   # 追踪阈值
    )

    # ==================== 打开摄像头 ====================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备是否已连接。")
        return

    print("已启动，按q退出")

    while True:
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败，程序退出。")
            break

        # 水平翻转，让显示效果像照镜子一样
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # OpenCV读进来的是BGR格式，MediaPipe需要RGB格式输入
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ==================== 结果处理与可视化 ====================
        if results.multi_hand_landmarks:
            total_fingers = 0      # 所有正常手势的手指总数
            has_offensive = False   # 是否检测到不雅手势

            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                # 获取当前手的左右标签
                label = results.multi_handedness[idx].classification[0].label

                # 分析每根手指的伸出状态
                finger_states = get_finger_states(hand_lm, label)
                finger_count = sum(finger_states)

                # ========== 竖中指检测 ==========
                if is_middle_finger_only(finger_states):
                    # 检测到竖中指 → 模糊手部区域 + 显示警告
                    has_offensive = True
                    frame = blur_hand_region(frame, hand_lm, h, w)

                    # 在手腕位置下方显示"BLOCKED"
                    wrist_x = int(hand_lm.landmark[0].x * w)
                    wrist_y = int(hand_lm.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        "BLOCKED",
                        (wrist_x - 50, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),  # 红色
                        2,
                    )
                else:
                    # 正常手势 → 画骨架 + 显示手指数
                    total_fingers += finger_count
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    # 在手腕下方标注 "左/右手: 手指数"
                    wrist_x = int(hand_lm.landmark[0].x * w)
                    wrist_y = int(hand_lm.landmark[0].y * h)
                    cv2.putText(
                        frame,
                        f"{label} Hand: {finger_count}",
                        (wrist_x - 40, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),  # 绿色
                        2,
                    )

            # 画面顶部的状态栏
            if has_offensive:
                # 检测到不雅手势时显示红色警告
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
                # 正常情况下显示手指总数
                cv2.putText(
                    frame,
                    f"Total Fingers: {total_fingers}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),  # 蓝色
                    3,
                )
        else:
            # 没检测到手的时候，显示提示信息
            cv2.putText(
                frame,
                "No hands detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        cv2.imshow("Hand Detection", frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==================== 清理资源 ====================
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
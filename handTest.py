# -*- coding: utf-8 -*-
"""
handTest.py - 基于肤色检测的手势识别（纯OpenCV实现）

功能说明：
    通过摄像头实时捕捉画面,利用HSV颜色空间提取肤色区域,
    再借助凸包和凸缺陷分析来估算伸出的手指数量。
    本程序不依赖深度学习模型，仅使用传统图像处理方法,
    适合作为手势识别的基础入门示例。

使用方式：
    直接运行本文件即可启动摄像头窗口，按 'q' 键退出。

依赖库：
    opencv-python, numpy
"""

import cv2
import numpy as np
import math


def main():
    # 打开默认摄像头（编号0）
    cap = cv2.VideoCapture(0)
    print("程序已启动，正在检测双手手势。按 'q' 键退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻转画面，这样用户看到的和镜子一样，操作更直观
        frame = cv2.flip(frame, 1)

        # ==================== 肤色检测 ====================
        # 把BGR图像转到HSV空间，方便按颜色范围提取皮肤区域
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 肤色在HSV空间中的大致范围（实际效果受光照影响较大）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 生成肤色二值掩膜，并做高斯模糊去噪
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # ==================== 轮廓检测 ====================
        # 在掩膜图上查找所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 按面积从大到小排序，取最大的两个轮廓作为候选手部区域
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            total_fingers = 0  # 两只手的手指总数
            hand_num = 0       # 检测到的手的数量

            # 遍历前两个最大的轮廓 (假设画面里最大的两个肤色块就是两只手)
            for contour in sorted_contours[:2]:

                # 轮廓面积太小的话，大概率是噪声，直接跳过
                if cv2.contourArea(contour) > 5000:
                    hand_num += 1

                    # ========== 凸包 & 凸缺陷分析 ==========
                    # 凸包：包住轮廓的最小凸多边形
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)  # 红色画轮廓
                    cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)     # 蓝色画凸包

                    # 获取凸包的索引形式，用于后续求凸缺陷
                    hull_indices = cv2.convexHull(contour, returnPoints=False)
                    defect_count = 0  # 有效凸缺陷数（约等于手指间的凹陷数）

                    # 至少需要4个凸包点才能计算凸缺陷
                    if hull_indices is not None and len(hull_indices) > 3:
                        try:
                            defects = cv2.convexityDefects(contour, hull_indices)
                            if defects is not None:
                                for i in range(defects.shape[0]):
                                    # s=起点索引, e=终点索引, f=最远点索引, d=距离
                                    s, e, f, d = defects[i, 0]
                                    start = tuple(contour[s][0])
                                    end = tuple(contour[e][0])
                                    far = tuple(contour[f][0])  # 凹陷最深处的点

                                    # 用余弦定理计算起点-最远点-终点构成的三角形夹角
                                    # 手指之间的夹角一般小于90度
                                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                                    # 防止分母为0
                                    if b * c != 0:
                                        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.29

                                        # 角度小于90度才认为是手指间的有效凹陷
                                        if angle <= 90:
                                            defect_count += 1
                                            # 在凹陷最深点画个黄色小圆标记
                                            cv2.circle(frame, far, 5, (0, 255, 255), -1)
                        except Exception as e:
                            # convexityDefects偶尔会抛异常，这里直接忽略
                            pass

                    # 手指数 = 凹陷数 + 1（比如4个手指缝对应5根手指）
                    # 但如果没有检测到凹陷，至少算1根手指（握拳的情况）
                    fingers = defect_count + 1 if defect_count > 0 else 1
                    total_fingers += fingers

                    # 在手部区域上方标注手指数
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.putText(frame, f"Hand: {fingers}", (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 左上角汇总显示：检测到几只手、总手指数
            cv2.putText(frame, f"Hands: {hand_num} | Total Fingers: {total_fingers}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 同时显示原始画面和肤色掩膜，方便调试
        cv2.imshow("Full Screen Dual Hand Detection", frame)
        cv2.imshow("Skin Mask", mask)

        # 按q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import math

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    print("程序已启动，正在全屏检测双手手势。按 'q' 键退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 翻转画面，符合照镜子的直觉
        frame = cv2.flip(frame, 1)
        
        # 1. 图像预处理与肤色提取
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 2. 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 【关键修改】：按面积从大到小排序轮廓
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            total_fingers = 0
            hands_detected = 0
            
            # 【关键修改】：遍历前两个最大的轮廓 (假设画面里最大的两个肤色块就是两只手)
            for contour in sorted_contours[:2]:
                
                # 面积阈值过滤干扰
                if cv2.contourArea(contour) > 5000:
                    hands_detected += 1
                    
                    # 3. 计算凸包和凸缺陷
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2) # 画轮廓 (红)
                    cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)   # 画凸包 (蓝)
                    
                    hull_indices = cv2.convexHull(contour, returnPoints=False)
                    defect_count = 0
                    
                    if hull_indices is not None and len(hull_indices) > 3:
                        try:
                            defects = cv2.convexityDefects(contour, hull_indices)
                            
                            if defects is not None:
                                for i in range(defects.shape[0]):
                                    s, e, f, d = defects[i, 0]
                                    start = tuple(contour[s][0])
                                    end = tuple(contour[e][0])
                                    far = tuple(contour[f][0])
                                    
                                    # 余弦定理计算角度
                                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                                    
                                    # 避免除以 0 的错误
                                    if b * c != 0:
                                        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57.29
                                        
                                        if angle <= 90:
                                            defect_count += 1
                                            cv2.circle(frame, far, 5, (0, 255, 255), -1)
                        except Exception as e:
                            pass
                            
                    # 计算当前这只手的手指数
                    fingers = defect_count + 1 if defect_count > 0 else 1
                    total_fingers += fingers
                    
                    # 【新增】：在每只手的光标上方显示当前手的手指数
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.putText(frame, f"Hand: {fingers}", (x, y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 左上角显示检测到的手的数量和总手指数
            cv2.putText(frame, f"Hands: {hands_detected} | Total Fingers: {total_fingers}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 显示画面
        cv2.imshow("Full Screen Dual Hand Detection", frame)
        cv2.imshow("Skin Mask", mask) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
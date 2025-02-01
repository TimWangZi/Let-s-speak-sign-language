import cv2 as cv
import mediapipe as mp

#调用手部检测的模块
mp_hand=mp.solutions.hands
#创建对象
hand=mp_hand.Hands()
#调用绘制和连接手部关键点的模块
mp_draw=mp.solutions.drawing_utils

#姿态检测
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
# 参数1：复杂程度，0，1，2 越大越精确，但反应也更慢
# 参数2：检测置信度：0~1 越大越精确，越小越灵敏
# 参数3：跟踪置信度：0~1 对检测到的目标跟踪的准确度，越大越精确，越小越灵敏

# 初始化MediaPipe面部关键点模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,  # 适用于动态图像
                max_num_faces=1,  # 最多检测1张脸
                refine_landmarks=True,  # 不细化关键点
                min_detection_confidence=0.5)
#打开摄像头
cap=cv.VideoCapture(0)

while cap.isOpened():
    #接受两个变量，前者控制打开的状态，后者接收每一帧
    cond,img=cap.read()
    #如果成功打开，继续执行
    if cond:
        #用opencv读取的图像格式是BGR，但mediapipe要求的是RGB,这里转一下
        img_RGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        #把窗口放大一点，太小了看着不舒服，可以调，但是要保证img和img_RGB的size一样
        img_RGB=cv.resize(img_RGB,(1080,960))
        img=cv.resize(img,(1080,960))
        #获得图像的h和w
        h,w,_=img_RGB.shape
        #print(type(img_RGB)) 输出结果确实是ndarray类型
        #----------------------手部检测----------------------
        #获得手部关键点的信息
        hands_result = hand.process(img_RGB)
        #获得手部关键点的坐标
        hands_loc=hands_result.multi_hand_landmarks
        #print(hands_loc)
        loc_list=[]
        #绘制出21个手部关键点并连接
        if hands_result.multi_hand_landmarks:
            for loc in hands_result.multi_hand_landmarks:
                draw_hands=mp_draw.draw_landmarks(img,
                                                       loc,
                                                       mp_hand.HAND_CONNECTIONS,
                                                       connection_drawing_spec=mp_draw.DrawingSpec((0,255,0),thickness=1))

        #-----------------------上肢检测---------------------
        #获取上肢关键点的信息
        pose_result=pose.process(img_RGB)
        if pose_result.pose_landmarks:
            landmarks=pose_result.pose_landmarks.landmark

            # 左肩、左肘、左腕的索引
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # 转换为像素坐标
            left_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            left_elbow = (int(left_elbow.x * w), int(left_elbow.y * h))
            left_wrist = (int(left_wrist.x * w), int(left_wrist.y * h))

            # 右肩、右肘、右腕的索引
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # 转换为像素坐标
            right_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            right_elbow = (int(right_elbow.x * w), int(right_elbow.y * h))
            right_wrist = (int(right_wrist.x * w), int(right_wrist.y * h))

            #将点连接成线
            cv.line(img, left_shoulder, left_elbow, (255, 0, 255), 2)
            cv.line(img, left_elbow, left_wrist, (255, 0, 255), 2)
            cv.line(img, right_shoulder, right_elbow, (255, 0, 255), 2)
            cv.line(img, right_elbow, right_wrist, (255, 0, 255), 2)
            #这里把整个上肢另外检测了一遍
            #mp_draw.draw_landmarks(img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #--------------------------面部检测-------------------------

        # 定义需要提取的9个关键点索引（根据MediaPipe的面部网格图）
        LANDMARK_INDEX = [
                33,  # 左眼内角
                133,  # 左眼外角
                362,  # 右眼内角
                263,  # 右眼外角
                1,  # 鼻尖
                61,  # 左嘴角
                291,  # 右嘴角
                0,  # 上唇中点
                17  # 下唇中点
        ]


        # 检测面部关键点
        results = face_mesh.process(img_RGB)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取并绘制关键点
                for idx in LANDMARK_INDEX:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)  # 转换为像素坐标
                    y = int(landmark.y * h)
                    cv.circle(img, (x, y), 2, (0, 255, 0), -1)




        cv.imshow("img_RGB",img)
        if cv.waitKey(1) and 0xFF==ord("q"):
            break

cap.release()
cv.destroyAllWindows()
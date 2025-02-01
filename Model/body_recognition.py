import cv2 as cv
import mediapipe as mp

class Body_recognition:
    def __init__(self):
        # 手部检测初始化
        self.mp_hand = mp.solutions.hands
        self.hand = self.mp_hand.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        # 姿态检测初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # 面部检测初始化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 定义面部需要提取的9个关键点索引
        self.LANDMARK_INDEX = [
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

    def detect(self):
        # 打开摄像头
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            # 接受两个变量，前者控制打开的状态，后者接收每一帧
            cond, img = cap.read()
            # 如果成功打开，继续执行
            if cond:
                # 用opencv读取的图像格式是BGR，但mediapipe要求的是RGB，这里转一下
                img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                # 把窗口放大一点，太小了看着不舒服，可以调，但是要保证img和img_RGB的size一样
                img_RGB = cv.resize(img_RGB, (1080, 960))
                img = cv.resize(img, (1080, 960))
                # 获得图像的h和w
                h, w, _ = img_RGB.shape

                # 手部检测
                hands_result = self.hand.process(img_RGB)
                hands_list = []
                hands_loc = hands_result.multi_hand_landmarks
                if hands_result.multi_hand_landmarks:
                    for loc in hands_result.multi_hand_landmarks:
                        for landmark in loc.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            hands_list.append((x, y))
                        self.mp_draw.draw_landmarks(img, loc, self.mp_hand.HAND_CONNECTIONS,
                                                    connection_drawing_spec=self.mp_draw.DrawingSpec((0, 255, 0), thickness=1))

                # 上肢检测
                pose_result = self.pose.process(img_RGB)
                upper_body_loc = []
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks.landmark

                    # 左肩、左肘、左腕的索引
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

                    # 转换为像素坐标
                    left_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                    left_elbow = (int(left_elbow.x * w), int(left_elbow.y * h))
                    left_wrist = (int(left_wrist.x * w), int(left_wrist.y * h))

                    # 右肩、右肘、右腕的索引
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                    # 转换为像素坐标
                    right_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                    right_elbow = (int(right_elbow.x * w), int(right_elbow.y * h))
                    right_wrist = (int(right_wrist.x * w), int(right_wrist.y * h))

                    # 将点连接成线
                    cv.line(img, left_shoulder, left_elbow, (255, 0, 255), 2)
                    cv.line(img, left_elbow, left_wrist, (255, 0, 255), 2)
                    cv.line(img, right_shoulder, right_elbow, (255, 0, 255), 2)
                    cv.line(img, right_elbow, right_wrist, (255, 0, 255), 2)

                    # 上肢关键点的坐标
                    upper_body_loc = [left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist]

                # 面部检测
                results = self.face_mesh.process(img_RGB)
                face_loc = []
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 提取并绘制关键点
                        for idx in self.LANDMARK_INDEX:
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)  # 转换为像素坐标
                            y = int(landmark.y * h)
                            face_loc.append((x, y))
                            cv.circle(img, (x, y), 2, (0, 255, 0), -1)

                cv.imshow("img_RGB", img)

                # 每一帧处理完后就返回该帧的坐标列表
                yield face_loc, upper_body_loc, hands_list

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv.destroyAllWindows()



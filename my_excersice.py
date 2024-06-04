import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils   #pozları görselleştirmek için kullanılan çizimler
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)
  

  if angle > 180.0:
    angle = 360-angle

  return angle



def HighKnees():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        #print("HIGH_KNEES")
        if right_knee < 0.8 and left_wrist < 0.4:
          
          stage = "left"
        
        if left_knee < 0.8 and right_wrist < 0.4 and stage == "left":
          stage = "right"
          counter +=1
          print(counter)
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()


def Punches():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


        left_bicep_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_bicep_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        #print(left_bicep_angle, right_bicep_angle)
        if right_bicep_angle < 30 and left_bicep_angle > 15:
            stage = "left"
            
        if left_bicep_angle < 30 and  right_bicep_angle > 100 and stage == "left":
            stage = "right"
            counter +=1
            print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()


def LegCurl():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark

        knee_cord = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle_cord = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        hip_cord = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        leg_curl_angle = calculate_angle(knee_cord,ankle_cord,hip_cord)
        if leg_curl_angle < 40:
          stage = "down"
      
        if leg_curl_angle > 70 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

def Squat():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark

        shoulder_height = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        if shoulder_height > 0.80:
          stage = "down"

        if shoulder_height < 0.65 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()


def Biceps():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        left_bicep_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
      

        if left_bicep_angle > 100:
          stage = "down"
        if left_bicep_angle < 50 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

def Plank():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark

        shoulder_height = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        if shoulder_height > 0.80:
          stage = "down"

        if shoulder_height < 0.65 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()



def Superman():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        if right_knee < 0.8 and left_wrist < 0.4:
          
          stage = "down"
        
        if left_knee < 0.8 and right_wrist < 0.4 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()


def CSqueeze():
  counter = 0
  stage = None
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
    while cap.isOpened():
      ret, frame = cap.read()

      
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Görüntü tespit kısmı
      results = pose.process(image)

      # görüntüyü işlemek için tekrar bgr'a döndürüyoruz
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # eklemleri çıkarma
      try:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        left_bicep_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
      

        if left_bicep_angle > 100:
          stage = "down"
        if left_bicep_angle < 50 and stage == "down":
          stage = "up"
          counter +=1
          print(counter)
        
        
      except:
        pass 


      cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

      cv2.putText(image, 'REPS', (15,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, str(counter),
                  (10,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
      
      cv2.putText(image, 'STAGE', (65,12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
      
      cv2.putText(image, stage,
                  (60,60),
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

      cv2.imshow("VIDEO FEED", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()



#Biceps()

#Squat()

#LegCurl()

#Punches()

#HighKnees()




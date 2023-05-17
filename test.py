import cv2
import mediapipe as mp
import pickle
import numpy as np

'''
# Load pickled model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)

# Extract model object
model = model_dict['model']

# Get number of features
num_features = model.n_features_in_
print("Number of features in the model:", num_features)
'''
# load the model from file
model_dict = pickle.load(open('./model2.p','rb'))
model = model_dict['model']
m=""
cap = cv2.VideoCapture(0)
ot=cv2.imread("images.png")
out=cv2.resize(ot,(800,400))
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
l=0
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

while True:

    data_aux = []
    x_ = []
    y_ = []
    z_ = [] # added for z-coordinate
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                x_.append(x)
                y_.append(y)
                z_.append(z) # add z-coordinate to the data_aux list

        # use the x_, y_, and z_ lists to calculate the bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # concatenate x_, y_, and z_ lists into a single numpy array
        data_aux = np.concatenate([x_, y_, z_], axis=0)
        # pad the array to match the expected number of features (84)
        data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), mode='constant')

        prediction = model.predict([data_aux])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 3, cv2.LINE_AA)
        k= cv2.waitKey(1)
        if(k==ord(' ')):
            if(l==1):
                cv2.destroyWindow("output")
            m=m+predicted_character
            print("String up to now predicted  :   ",m )
            cv2.putText(out, "String up to now  :" + m, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            l=1
            cv2.imshow("output",out)
        elif(k==ord('q')):
            break
    frame = cv2.resize(frame, (400, 400))
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

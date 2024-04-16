import cv2
import math
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import matplotlib.gridspec as gridspec
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation


def detect_plate(img, text=''): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()

    plate_cascade = cv2.CascadeClassifier('../weights/indian_license_plate.xml') # loading the trained model for license plate detection.
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,181,155), 3) 
    
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (51,181,155), 1, cv2.LINE_AA)
        return plate_img, plate

    cv2.imwrite('../images/output/plate.jpg', plate)
    st.image('../images/output/plate.jpg', caption='Detected License Plate', use_column_width=True)
    return plate_img, plate

# Match contours to license plate or character template
def find_contours(dimensions, img):

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('../images/output/contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
    cv2.imwrite('../images/output/contour.jpg', ii)
    st.image('../images/output/contour.jpg', caption='Contour Image', use_column_width=True)
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character output according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting output
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    cv2.imwrite('../images/output/contour.jpg',img_binary_lp)
    # st.image('../images/contour.jpg', caption='Contour Image.', use_column_width=True)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def print_char(char):
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(char[i], cmap='gray')
        plt.axis('off')
    plt.savefig('../images/output/characters.jpg')
    st.image('../images/output/characters.jpg', caption='Characters', use_column_width=True)

def load_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))

    # Then, load the saved weights
    model.load_weights('../weights/license_plate_model_weights.h5')

    return model

# Predicting the output
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        model = load_model()
        y_ = model.predict(img)[0] #predicting the class

        y_scalar = np.argmax(y_)
        character = dic[y_scalar] # get the class
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

def print_predicted_char(char):
    # Segmented characters and their predicted value.
    plt.figure(figsize=(10,6))
    for i,ch in enumerate(char):
        img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        plt.subplot(3,4,i+1)
        plt.imshow(img,cmap='gray')
        plt.title(f'predicted: {show_results(char)[i]}')
        plt.axis('off')
    plt.savefig('../images/output/predicted_characters.jpg')
    st.image('../images/output/predicted_characters.jpg', caption='Predicted Characters', use_column_width=True)

def save_predicted_image():
    img = cv2.imread('../images/output/input.jpg')
    output_img, plate = detect_plate(img)
    char = segment_characters(plate)
    print_predicted_char(char)
    plate_number = show_results(char)
    plate_number = str(plate_number)
    output_img, plate = detect_plate(img, plate_number)
    cv2.imwrite('../images/output/output.jpg', output_img)

def main():
    st.title("License Plate Detection")
    st.write("This is a simple web app to detect license plates in output")

    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

    if uploaded_file is not None:
        with open('../images/output/input.jpg', 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.image('../images/output/input.jpg', caption='Uploaded Image', use_column_width=True)
        save_predicted_image()
        st.image('../images/output/output.jpg', caption='Predicted Image', use_column_width=True)

if __name__ == '__main__':
    main()

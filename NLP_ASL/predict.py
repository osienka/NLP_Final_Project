import numpy as np
from keras.models import load_model
from keras.utils import load_img


def decode(pred):
    pred = [chr(i + ord('a') - 10) if i >= 10 else str(i) for i in pred]
    res = ""
    return res.join(pred)


model_new = load_model("./model.h5")


def entering_images(pred):
    path = input("Please enter the path to the image you want to use. ")
    try:
        img = load_img(path, color_mode='grayscale', target_size=(100, 100))
        x = np.array(img) / 255
        x = np.array(x).reshape(-1, 100, 100, 1)
        pred.append(decode(model_new.predict(x).argmax(axis=-1)))
    except:
        print("Error. Please make sure to enter correct path")
    return pred


entry = input("Do you want to enter an image? Please write 'yes' or 'no'.")
t = True
predictions = []
while t:
    if entry == 'yes':
        entering_images(predictions)
        if len(predictions) > 0:
            print("Those are already predicted values: ", *predictions)
        entry = input("Do you want to enter an image? Please write 'yes' or 'no'.")
    elif entry == 'no':
        print("Predicted values: ", *predictions)
        t = False
    else:
        print("An error has occured. Please try again.")
        entry = input("Do you want to enter an image? Please write 'yes' or 'no'.")
        if len(predictions) > 0:
            print("Those are already predicted values: ", *predictions)


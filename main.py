import pickle
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
import os


knn_from_joblib = joblib.load('filename.pkl')  
img = Image.open(input("Enter Image name")) #add your image to the current directory and then input when prompted
im2arr = np.array(img)
lol,lmao=im2arr.shape
im2arr=im2arr.reshape(lol*lmao)
resu=knn_from_joblib.predict([im2arr])
print(resu)


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import itertools
from collections import defaultdict
def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]

def predict(img_path : str, model: Model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

def findDifference(f1, f2):
    return np.linalg.norm(f1-f2)

def findDifferences(feature_vectors_train,feature_vectors_test):
    similar = defaultdict(list)
    keys_train = [k for k,v in feature_vectors_train.items()]
    keys_test = [k for k,v in feature_vectors_test.items()]
    mini =  {}
    for k in keys_test:
        mini[k] = float('inf')
    possible_combinations=list()
    for i in keys_test:
      for j in keys_train:
        possible_combinations.append([i,j])
    print('Analyzing All combinations of Objects: ')
    for k,v in possible_combinations:
       diff=findDifference(feature_vectors_test[k],feature_vectors_train[v])
       
       '''if(diff<mini[k]):
                                mini[k] = diff'''
       similar[k].append([v,diff])
       similar[k].sort( key = lambda x:x[1])
    return similar 

def driver():
    print('--Welcome to Shopify Similarity Image Searching--')
    print('--Upload the image in uploads folder (Currentlt, Dog or Cat)--')
    feature_vectors_train: dict = {}
    feature_vectors_test: dict = {}
    top = int(input('How many Similar Images you want to see ?'))
    model = ResNet50(weights='imagenet')
    for img_path in getAllFilesInDirectory("images"):
        feature_vectors_train[img_path] = predict(img_path,model)[0]
    if top > len(feature_vectors_train):
      top = len(feature_vectors_train)
    for img_path in getAllFilesInDirectory("uploads"):
        feature_vectors_test[img_path] = predict(img_path,model)[0]
    results=findDifferences(feature_vectors_train,feature_vectors_test)
    for k,v in results.items():
        name = k.split('/')[-1]
        print(f'The Top {top} Objects to which {name} is Similar are: '+','.join(i.split('/')[-1] for i,j in v[:top]))
        print()    
    #print('Predicted:', decode_predictions(preds, top=3)[0])

driver()

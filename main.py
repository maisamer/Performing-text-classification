from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import re
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
N=1000

negativetexts=glob.glob("C:/Users/go/Downloads/txt_sentoken/neg/*.txt")
postivetexts=glob.glob("C:/Users/go/Downloads/txt_sentoken/pos/*.txt")
finalweightsPsotive={}
finalweightsNegative={}
cleannegtive=[]
cleanpostive=[]
cleannegtive2=[]
cleanpostive2=[]
def readFileneg(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        cleannegtive.append(lines)
        lines=re.split(' ',lines)
        lines.remove('')
        cleannegtive2.append(lines)
def readFilepos(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        cleanpostive.append(lines)
        lines=re.split(' ',lines)
        lines.remove('')
        cleanpostive2.append(lines)       
def getAllwords():
    for t in negativetexts:
          readFileneg(t)
    for t in postivetexts:
          readFilepos(t)
def maketdidf(ar):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(ar)
    idf = vectorizer.idf_ 
    finalwords=dict(zip(vectorizer.get_feature_names(), idf))
    return finalwords
def makeSets():
    x=1
    for i in cleannegtive2:
        se=set(i)
        x=x+1
        aN.append(se)
    for j in cleanpostive2:
        se=set(j)
        aP.append(se)
def makeFeatures():
    for x in aN:
        q=[]
        for w in finalweightsNegative:
            if w in x:
                q.append(finalweightsNegative[w])
            else:
                q.append(0.0)
        for w in finalweightsPsotive:
            q.append(0.0)     
        features.append(q)
        goals.append(0)
    for x in aP:
        q=[]
        for w in finalweightsNegative:
                q.append(0.0) 
        for w in finalweightsPsotive:
            if w in x:
                    q.append(finalweightsPsotive[w])
            else:
                q.append(0.0)       
        features.append(q)
        goals.append(1)  
def accuracy(goal_test,goal_predict):
    correct = 0
    for i,j in zip(goal_test,goal_predict) :  
        if i == j:
            correct += 1
    accuracy = float(correct)/len(goal_test)  #accuracy 
    return accuracy         
getAllwords()  #all texts (cleaned)
finalweightsNegative=maketdidf(cleannegtive)
finalweightsPsotive=maketdidf(cleanpostive)
bigsetsP={}
bigsetN={}
aN=[]
aP=[]
makeSets()
bigsetN=aN[0]
for i in aN:
      bigsetN=bigsetN.union(i)
bigsetsP=aP[0]
for i in aP:
      bigsetsP=bigsetsP.union(i) 
import numpy as np
goals=[]
features=[]
makeFeatures()
print("size features ",len(features))
print("//////////////////////////////////////////////// \n")
XX=[]
for i in features:
    mX=max(i)
    mI=min(i)
    s=[]
    for j in i:
        n = (j-mI)/(mX-mI)
        s.append(n)
    XX.append(s)
# reduce feature
pca = PCA(n_components=2 )
XX = pca.fit_transform(XX)
def randomInt (low , high) :
    return random.randint(low,high)
def randomFloat (low , high):
    return random.uniform(low, high)
from sklearn.cross_validation import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(XX,goals,train_size =0.7,random_state=1)
from sklearn.linear_model import LogisticRegression
itr = randomInt(100,1500)
t = randomFloat(0000.1,0.01)
logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=itr, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=1, solver='liblinear', tol=t,
          verbose=0, warm_start=False)


logistic.fit(train_set, goal_train)
pred=logistic.predict(test_set)
print("//////////////////////////////////////////////// \n")
print("Logistic_Regression algorithm accuracy is : %f" %(accuracy(goal_test,pred)))
print("//////////////////////////////////////////////// \n")
def readFileneg2(Filename):
        f = open(Filename,"r")
        lines = f.read()
        lines=lines.lower()
        lines=re.sub('[^A-Za-z]+',' ',  lines)
        lines=re.split(' ',lines)
        lines.remove('')
        testfile.append(lines)

testfile=[]
readFileneg2("cv004_12641.txt")
features2=[]
for i in finalweightsNegative:
    if i in testfile[0]:
        features2.append(finalweightsNegative[i])
    else:
        features2.append(0)
for i in finalweightsPsotive:
    if i in testfile[0]:
        features2.append(finalweightsPsotive [i])
    else:
        features2.append(0)        
#Normalized 
X=[]
maxx=max(features2)
minn=min(features2)
for i in features2:
    n = (i-minn)/(maxx-minn)
    X.append(n)    
ih=[]
ih.append(X)
ih = pca.transform(ih)
predtext=logistic.predict(ih)
print("//////////////////////////////////////////////// \n")
print("predict ",predtext)
print("//////////////////////////////////////////////// \n")

# Visualising the Training set results

X_set, y_set = test_set, goal_test
X_set = np.array(X_set)
y_set = np.array(y_set)
X1, X2= np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.05),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.05))
plt.contourf(X1, X2, logistic.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Performing text classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
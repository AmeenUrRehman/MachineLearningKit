
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# As we have multiple data and can have features more than 2 dimentions so to be in same dimensions we use PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("All in one Machine Learning Algoithms ")
st.write("""
##  Exploring different Machine Learning Classifier 
### Which one is the best Classifer? Choice is yours
""")

dataset_name = st.sidebar.selectbox("Select DataSets", ("Iris", "Breast Cancer", "Wine DataSets"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("K-Nearest Neighbor", "Support Vector Machine", "Random Forest Classifier"))

st.write(dataset_name)

# Defining a function for loading datasets
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target

    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of Datasets : ", X.shape)
st.write("Number of Classes : ", len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == "K-Nearest Neighbor":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C" , 0.0, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth" , 2 , 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "K-Nearest Neighbor":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Support Vector Machine":
        clf = SVC(C =params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"] , max_depth= params["max_depth"] , random_state= 1234)

    return clf

clf = get_classifier(classifier_name , params)

# Classification
X_train , X_test , y_train , y_test = train_test_split(X, y , random_state= 42 , test_size= 0.3)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test , y_pred)

st.write(f"Classifier name = {classifier_name}")
st.write(f"Accuracy =  {acc}")

# Plotting
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1 , x2, c = y, alpha= 0.8, cmap="viridis")
plt.xlabel("Principal Component Analysis 1")
plt.ylabel("Principal Component Analysis 2")
plt.colorbar()

# Show
st.pyplot(fig)
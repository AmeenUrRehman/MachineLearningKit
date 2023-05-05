
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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


st.title("All in one Machine Learning Algoithms ")
st.write("""
##  Exploring different Machine Learning Classifier 
### Which one is the best Classifer? Choice is yours
""")

dataset_name = st.sidebar.selectbox("Select DataSets", ("Iris", "Breast Cancer", "Wine DataSets"))

typeofAlgo = st.sidebar.selectbox("Select type of Algorithm" , ("Supervised Algorithm" , "Unsupervised Algorithm"))

SupervisedTypeCat = st.sidebar.selectbox("Select Supervised Classifier Categories",
                                         ("Logistic Regression" , "K-Nearest Neighbor", "Support Vector Machine",
                                         "Random Forest Classifier", "Linear Regression"))



classifier_name = st.sidebar.selectbox("Select Classifier", ("K-Nearest Neighbor", "Support Vector Machine", "Random Forest Classifier"))

st.write(dataset_name)

def type_algo(typeofAlgorithm):
    if typeofAlgorithm == "Supervised Algorithm":
        return get_SuperAlgo(SupervisedTypeCat)

def add_parameter(clf_name):
    params = dict()
    if clf_name == "K-Nearest Neighbor":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C" , 0.1, 10.0)
        params["C"] = C

    elif clf_name == "Logistic Regression":
        R = st.sidebar.slider("R", 1, 100)
    else:
        max_depth = st.sidebar.slider("max_depth" , 2 , 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter(classifier_name)

def get_SuperAlgo(SupervisedTypeCat, params):
    if SupervisedTypeCat == "K-Nearest Neighbor":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif SupervisedTypeCat == "Support Vector Machine":
        clf = SVC(C =params["C"])
    elif SupervisedTypeCat == "Linear Regression":
        clf = LinearRegression()
    elif SupervisedTypeCat == "Logistic Regression":
        clf = LogisticRegression(R = params["R"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=1234)
    return clf

clf = get_SuperAlgo(SupervisedTypeCat, params)


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

st.markdown("""
<style>
.reportview-container .markdown-text-container {
    font-family: cursive;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-family: cursive;
}
[class^="st-b"]  {
    color: white;
    font-family: Georgia, serif;
}
# .st-bb {
#     background-color: transparent;
# }
.st-at {
    background-color: #0c0080;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style> """
,
    unsafe_allow_html=True,
)
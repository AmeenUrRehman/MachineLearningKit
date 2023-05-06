import streamlit as st
import sklearn
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Machine Learning",
)

pg_bg_img = """
<style> 
[data-testid = "stAppViewContainer"]{
background-image: url("https://w0.peakpx.com/wallpaper/369/751/HD-wallpaper-mountain-landscape-sunset-minimalist-1-minimalism-minimalist-mountains-landscape-1-1-abstract-vector-deviantart-sunset.jpg");
background-size: cover;
}

[data-testid = "stToolbar"]{
right: 2rem}

[data-testid = "stMarkdownContainer"]{
font-size: 1rem;
font-family:fangsong;
}

[data-testid = "stHeader"]{
 background-color: rgba(0,0,0,0); 
 length: 1rem;
}

[data-testid = "stSidebar"]{
background-image: url("https://e0.pxfuel.com/wallpapers/278/784/desktop-wallpaper-latest-high-quality-iphone-11-background-for-everyone-designbolts-iphone-11-pink-thumbnail.jpg");
background-size: cover;
font-family: fangsong;
}

</style>
"""
st.markdown(pg_bg_img, unsafe_allow_html=True)

header1 = '<p style="font-family:fangsong ; color: White; font-size: 50px;"> <b>Multiple Machine Learning Algoithms at one place </b> </p>'
st.markdown(header1, unsafe_allow_html=True)

header2 = '<p style="font-family:fangsong; color: White; font-size: 30px;"> Exploring different Machine Learning Classifier</p>'
st.markdown(header2, unsafe_allow_html=True)

header3 = '<p style="font-family:Georgia, serif; color: White; font-size: 25px;">  Which one is the best Classifer? </p>'
st.markdown(header3, unsafe_allow_html=True)

st.sidebar.header("Customized Parameters")

dataset_name = st.sidebar.selectbox("Select Datasets", ("Iris", "Breast Cancer", "Wine DataSets"))

typeofAlgo = st.sidebar.selectbox("Select ML-Algorithm" , ("Supervised Algorithm" , "Unsupervised Algorithm"))


st.write(dataset_name)


# Supervised Machine Learning Algorithm
def add_parameter_Super(clf_name):
    params = dict()
    if clf_name == "K-Nearest Neighbor":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C" , 0.1, 10.0)
        params["C"] = C

    elif clf_name == "Logistic Regression":
        R = st.sidebar.slider("Max_Iteration", 1, 100)
        params["R"] = R
    elif clf_name == "Decision Tree":
        max_depth_decision = st.sidebar.slider("max_depth" , 2 , 15)
        params["max_depth_decision"] = max_depth_decision
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        params["criterion"] = criterion

    else:
        max_depth = st.sidebar.slider("max_depth" , 2 , 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_SuperAlgo(SupervisedTypeCat, params):
    if SupervisedTypeCat == "K-Nearest Neighbor":
        clf_Super = KNeighborsClassifier(n_neighbors=params["K"])
    elif SupervisedTypeCat == "Support Vector Machine":
        clf_Super = SVC(C =params["C"])
    elif SupervisedTypeCat == "Logistic Regression":
        clf_Super = LogisticRegression(max_iter= params["R"])
    elif SupervisedTypeCat == "Decision Tree":
        clf_Super = DecisionTreeClassifier(criterion= params["criterion"], max_depth= params["max_depth_decision"],
                                     random_state=1234)
    else:
        clf_Super = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=1234)
    return clf_Super


# Unsupervised Machine Learning Algorithms
def add_parameter_Unsuper(clf_name):
    params_unsupervised = dict()
    if clf_name == "K-Means Clustering":
        n_clusters = st.sidebar.slider("n_cluster", 1, 5)
        params_unsupervised["n_clusters"] = n_clusters
    return params_unsupervised

def get_UnSuperAlgo(UnSupervisedTypeCat, params_unsupervised):
    if UnSupervisedTypeCat == "K-Means Clustering":
        clf_unsuper = KMeans(n_clusters= params_unsupervised["n_clusters"])
    return clf_unsuper



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


def train_supervised(X,y, Classifer_Algo_super, Classifier_name_super):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    Classifer_Algo_super.fit(X_train, y_train)
    y_pred = Classifer_Algo_super.predict(X_test)

    Accuracy_super = accuracy_score(y_test, y_pred)

    st.write(f"Classifier name = {Classifier_name_super}")
    st.write(f"Accuracy =  {Accuracy_super}")

    # Plotting
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    # Show
    st.pyplot(fig)

def train_unsupervised(X,y, Classifer_Algo_unsuper, Classifier_name_unsuper):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    Classifer_Algo_unsuper.fit(X_train)
    y_pred = Classifer_Algo_unsuper.predict(X_test)

    Accuracy_unsuper = accuracy_score(y_test, y_pred)

    st.write(f"Classifier name = {Classifier_name_unsuper}")
    st.write(f"Accuracy =  {Accuracy_unsuper}")

    # Plotting
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()

    # Show
    st.pyplot(fig)

def type_algo(typeofAlgorithm):
    if typeofAlgorithm == "Supervised Algorithm":
        SupervisedTypeCat = st.sidebar.selectbox("Select Supervised Classifier Categories",
                                                 ("Logistic Regression", "K-Nearest Neighbor", "Support Vector Machine",
                                                  "Random Forest Classifier", "Decision Tree"))
        params = add_parameter_Super(SupervisedTypeCat)
        clf_Super = get_SuperAlgo(SupervisedTypeCat, params)
        trainsuper = train_supervised(X,y, clf_Super, SupervisedTypeCat)


    else:
        UnSupervisedTypeCat = st.sidebar.selectbox("Select Unsupervised Classifier Categories",
                                                   ("K-Means Clustering", "Hierarchial Clustering"))
        params_unsuper = add_parameter_Unsuper(UnSupervisedTypeCat)
        clf_unsuper = get_UnSuperAlgo(UnSupervisedTypeCat, params_unsuper)
        trainunsuper = train_unsupervised(X,y, clf_unsuper, UnSupervisedTypeCat)


AlgotithmSelected = type_algo(typeofAlgo)


st.markdown("""
<style>

.reportview-container .markdown-text-container {
    font-family:  Georgia, serif;
    
}
.sidebar .sidebar-content {

    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
    
}

.sidebar .sidebar-selectbox{
    font-family:  Georgia, serif;    

}
.Widget>label {
    color: white;
    font-family:  Georgia, serif;
    
}
[class^="st-b"]  {
    color: white;
    font-family: Georgia, serif;
}

</style> """
,
    unsafe_allow_html=True,
)



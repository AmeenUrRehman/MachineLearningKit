import streamlit as st

st.title("All in one Machine Learning Algoithms ")
st.write("""
##  Exploring different Machine Learning Classifier 
### Which one is the best Classifer? Choice is yours
""")

dataset_name = st.sidebar.selectbox("Select DataSets" , ("Iris" , "Breast Cancer", "Wine DataSets"))
st.write(dataset_name)
st.write("Done ")
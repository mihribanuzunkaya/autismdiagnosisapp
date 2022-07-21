
import numpy as np 
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from streamlit_shap import st_shap
import shap 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle 
import seaborn as sns
import streamlit.components.v1 as components

import eli5
from eli5 import formatters
from IPython.display import display

from matplotlib import pyplot as plt
import pandas as pd
from pyrsistent import s
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle
import numpy as np
import lime
import lime.lime_tabular
from streamlit import components

import warnings


warnings.filterwarnings("ignore")

import streamlit as st 


st.sidebar.title("Predict or Explore")
page = st.sidebar.selectbox("Choose the page you want to use",("Predict", "Explore"))

if page == "Predict":

    st.set_option('deprecation.showPyplotGlobalUse', False)

    s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv')
    s_all.drop(['Participant'],axis=1, inplace= True)


    X=s_all.drop(['Class'] , axis=1) 
    y=s_all['Class']




    X=s_all.drop(['Class'] , axis=1) 
    y=s_all['Class']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

    def load_model():
        with open ('knn_allselected.pkl', 'rb') as file :
            data = pickle.load(file)
        return data    

    data = load_model()

    knn_grid = data ["knn"]

    def load_model():
        with open ('mlp_allselected.pkl', 'rb') as file :
            data = pickle.load(file)
        return data    

    data = load_model()

    mlp_grid = data ["mlp"]

    def load_model():
        with open ('svm_allselected.pkl', 'rb') as file :
            data = pickle.load(file)
        return data    

    data = load_model()

    svm_grid = data ["svm"]

    def load_model():
        with open ('rf_allselected.pkl', 'rb') as file :
            data = pickle.load(file)
        return data    

    data = load_model()

    rfc_grid = data ["rf"]

    def load_model():
        with open ('dt_allselected.pkl', 'rb') as file :
            data = pickle.load(file)
        return data    

    data = load_model()

    dtree_grid = data ["dt"]

    
    st.title("XAI Based Autism Spectrum Disorder Diagnosis Web App")

    st.write ("""### Please fill the given features below""")
        
    a18 = st.number_input("a18=Action Net Dwell Time [ms]",format="%.2f")
    a19 = st.number_input("a19=Action Dwell Time [ms]",format="%.2f" )
    a20 = st.number_input("a20=Action Glance Duration [ms]",step=0.01,format="%.2f" )
    a21 = st.number_input("a21=Action Diversion Duration [ms]",step=0.01,format="%.2f" )
    a25 = st.number_input("a25=Action Fixation Count",step=0.01,format="%.2f" )
    a26 = st.number_input("a26=Action Net Dwell Time [%]",step=0.01,format="%.2f" )
    a27 = st.number_input("a27=Action Dwell Time [%]",step=0.01,format="%.2f" )
    a28 = st.number_input("a28=Action Fixation Time [ms]",step=0.01,format="%.2f" )
    a29 = st.number_input("a29=Action Fixation Time [%]",step=0.01,format="%.2f" )
    f16 = st.number_input("f16=Face Entry Time [ms]",step=0.01,format="%.2f" )
    f17 = st.number_input("f17=Face Sequence",step=0.01,format="%.2f" )
    f18 = st.number_input("f18=Face Net Dwell Time [ms]",step=0.01,format="%.2f" )
    f19 = st.number_input("f19=Face Dwell Time [ms]",step=0.01,format="%.2f" )
    f20 = st.number_input("f20=Face Glance Duration [ms]",step=0.01,format="%.2f" )
    f21 = st.number_input("f21=Face Diversion Duration [ms]",step=0.01,format="%.2f" )
    f22 = st.number_input("f22=Face First Fixation Duration [ms]",step=0.01,format="%.2f" )
    f23 = st.number_input("f23=Face Glances Count",step=0.01,format="%.2f" )
    f24 = st.number_input("f24=Face Revisits",step=0.01,format="%.2f" )
    f25 = st.number_input("f25=Face Fixation Count",step=0.01,format="%.2f" )
    f26 = st.number_input("f26=Face Net Dwell Time [%]",step=0.01,format="%.2f" )
    f27 = st.number_input("f27=Face Dwell Time [%]",step=0.01,format="%.2f" )
    f28 = st.number_input("f28=Face Fixation Time [ms]",step=0.01,format="%.2f" )
    f29 = st.number_input("f29=Face Fixation Time [%]",step=0.01,format="%.2f" )
    f30 = st.number_input("f30= Face Average Fixation Duration [ms]",step=0.01,format="%.2f" )
    FixationCount = st.number_input("Event Fixation Time",step=0.01,format="%.2f")
    FixationDurationAverage = st.number_input("Event Fixation Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationAverage = st.number_input("Event Saccade Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationMaximum = st.number_input("Event Saccade Duration Maximum",step=0.01,format="%.2f" )
    SaccadeAmplitudeMinimum = st.number_input("Event Saccade Amplitude Minimum",step=0.01,format="%.2f" )
    SaccadeLatencyAverage = st.number_input("Event Saccade Latency Average [ms]",step=0.01,format="%.2f" )
    BlinkDurationTotal = st.number_input("Event Blink Duration Total [ms]",step=0.01,format="%.2f")

    classifier_name = st.selectbox("Select Classifer",("KNN", "SVM", "MLP","Random Forest","Decision Tree"))
    ok = st.button("Make Prediction")

    if ok :

            
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)

        if classifier_name == "KNN":
            prediction = knn_grid.predict(X)
            prediction_proba = knn_grid.predict_proba(X)
            st.subheader("The KNN Prediction is")
            st.write(prediction)
            st.subheader("KNN Prediction Probability")
            st.write(prediction_proba)

        elif classifier_name == "SVM" :
            prediction = svm_grid.predict(X)
            prediction_proba = svm_grid.predict_proba(X)
            st.subheader("The SVM Prediction is")
            st.write(prediction)
            st.subheader("SVM Prediction Probability")
            st.write(prediction_proba)
            

        elif classifier_name == "Random Forest":
            prediction = rfc_grid.predict (X)
            prediction_proba = rfc_grid.predict_proba(X) 
            st.subheader("The Random Forest Prediction is")
            st.write(prediction)
            st.subheader("Random Forest Prediction Probability")
            st.write(prediction_proba)

        elif classifier_name == "MLP":
            prediction = mlp_grid.predict(X)  
            prediction_proba = mlp_grid.predict_proba(X) 
            st.subheader("The MLP Prediction is")
            st.write(prediction)
            st.subheader("MLP Prediction Probability")
            st.write(prediction_proba)  

        else :
            prediction = dtree_grid.predict(X)
            prediction_proba = dtree_grid.predict_proba(X) 
            st.subheader("The Decision Tree Prediction is")
            st.write(prediction)
            st.subheader("Decision Tree Prediction Probability")
            st.write(prediction_proba)
                    
    if st.checkbox("LIME Explanation") :
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)
        if classifier_name == "KNN":
                
            df = s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
                

            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
                
            lime_exp= lime_explainer.explain_instance(data_row = np.array(feature_list), predict_fn = knn_grid.predict_proba, num_features=33)

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)

        elif classifier_name == "SVM":
            df = s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
                

            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
                
            lime_exp= lime_explainer.explain_instance(data_row = np.array(feature_list), predict_fn = svm_grid.predict_proba, num_features=33)

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)

                
        elif classifier_name == "MLP":
                
                df = s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv') 
                X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
                X = X.astype(float)
                feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

                feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
                class_names = ['NG', 'OSB']

                import lime
                from lime import lime_tabular
                
                lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
                lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = mlp_grid.predict_proba, num_features=33, )

                html_lime = lime_exp.as_html()
                components.v1.html(html_lime, width=1100, height=350, scrolling=True)
        elif classifier_name=="Random Forest":
                

                df = s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv') 
                X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
                X = X.astype(float)
                feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

                feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
                class_names = ['NG', 'OSB']

                import lime
                from lime import lime_tabular
                
                lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
                lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = rfc_grid.predict_proba, num_features=33, )

                html_lime = lime_exp.as_html()
                components.v1.html(html_lime, width=1100, height=350, scrolling=True)
        
        else :
                df = s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv') 
                X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
                X = X.astype(float)
                feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

                feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
                class_names = ['NG', 'OSB']

                import lime
                from lime import lime_tabular
                
                lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
                lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = dtree_grid.predict_proba, num_features=33, )

                html_lime = lime_exp.as_html()
                components.v1.html(html_lime, width=1100, height=350, scrolling=True)

    st.write("$Citation Request: Kok, I., Uzunkaya, M., Ozdemir, S., Bulbul, I. A., & Ozdemir, S. (2022). XAI-based Early Autism Diagnostic Interface.$ [Click for Document](https://aiot.cs.hacettepe.edu.tr/etjasd/XAI_Based_Early_Autism_Diognostic_Interface.pdf)")

else:

    import numpy as np
    from sklearn.model_selection import train_test_split
    import streamlit as st
    import pandas as pd
    from streamlit_shap import st_shap
    import shap 
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import pickle 
    import seaborn as sns
    import streamlit.components.v1 as components
    import streamlit as st
    import eli5
    from eli5 import formatters
    from IPython.display import display
    s_all = pd.read_csv('https://raw.githubusercontent.com/mihribanuzunkaya/autismdiagnosisapp/main/3_xAll_AFB_Event_selected.csv')
    s_all.drop(['Participant'],axis=1, inplace= True)
    X=s_all.drop(['Class'] , axis=1) 
    y=s_all['Class']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    st.title("XAI Based Autism Spectrum Disorder Diagnosis Web App")
    st.sidebar.write ("""### Please fill the given features below""")
    
    a18 = st.number_input("a18=Action Net Dwell Time [ms]",format="%.2f")
    a19 = st.number_input("a19=Action Dwell Time [ms]",format="%.2f" )
    a20 = st.number_input("a20=Action Glance Duration [ms]",step=0.01,format="%.2f" )
    a21 = st.number_input("a21=Action Diversion Duration [ms]",step=0.01,format="%.2f" )
    a25 = st.number_input("a25=Action Fixation Count",step=0.01,format="%.2f" )
    a26 = st.number_input("a26=Action Net Dwell Time [%]",step=0.01,format="%.2f" )
    a27 = st.number_input("a27=Action Dwell Time [%]",step=0.01,format="%.2f" )
    a28 = st.number_input("a28=Action Fixation Time [ms]",step=0.01,format="%.2f" )
    a29 = st.number_input("a29=Action Fixation Time [%]",step=0.01,format="%.2f" )
    f16 = st.number_input("f16=Face Entry Time [ms]",step=0.01,format="%.2f" )
    f17 = st.number_input("f17=Face Sequence",step=0.01,format="%.2f" )
    f18 = st.number_input("f18=Face Net Dwell Time [ms]",step=0.01,format="%.2f" )
    f19 = st.number_input("f19=Face Dwell Time [ms]",step=0.01,format="%.2f" )
    f20 = st.number_input("f20=Face Glance Duration [ms]",step=0.01,format="%.2f" )
    f21 = st.number_input("f21=Face Diversion Duration [ms]",step=0.01,format="%.2f" )
    f22 = st.number_input("f22=Face First Fixation Duration [ms]",step=0.01,format="%.2f" )
    f23 = st.number_input("f23=Face Glances Count",step=0.01,format="%.2f" )
    f24 = st.number_input("f24=Face Revisits",step=0.01,format="%.2f" )
    f25 = st.number_input("f25=Face Fixation Count",step=0.01,format="%.2f" )
    f26 = st.number_input("f26=Face Net Dwell Time [%]",step=0.01,format="%.2f" )
    f27 = st.number_input("f27=Face Dwell Time [%]",step=0.01,format="%.2f" )
    f28 = st.number_input("f28=Face Fixation Time [ms]",step=0.01,format="%.2f" )
    f29 = st.number_input("f29=Face Fixation Time [%]",step=0.01,format="%.2f" )
    f30 = st.number_input("f30= Face Average Fixation Duration [ms]",step=0.01,format="%.2f" )
    FixationCount = st.number_input("Event Fixation Time",step=0.01,format="%.2f")
    FixationDurationAverage = st.number_input("Event Fixation Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationAverage = st.number_input("Event Saccade Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationMaximum = st.number_input("Event Saccade Duration Maximum",step=0.01,format="%.2f" )
    SaccadeAmplitudeMinimum = st.number_input("Event Saccade Amplitude Minimum",step=0.01,format="%.2f" )
    SaccadeLatencyAverage = st.number_input("Event Saccade Latency Average [ms]",step=0.01,format="%.2f" )
    BlinkDurationTotal = st.number_input("Event Blink Duration Total [ms]",step=0.01,format="%.2f")

    

    ok = st.button("See Random Forest Prediction")

    if ok :
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)

        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)
        st.subheader("Random Forest Prediction")
        st.write(prediction)
        st.subheader(" Random Forest Prediction Probability")
        st.write(prediction_proba)

        st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.checkbox("SHAP & ELI5 Explanation") :  
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)  
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)
        
        st.title("SHAP Explanation")
        explainer = shap.TreeExplainer(model)
        shap_values= explainer.shap_values(X_test)
        
        
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X_test),400)

        st.pyplot(shap.summary_plot(shap_values[1], X_test))

        feature_names=X_train.columns.values
        class_names = ["NG" , "OSB"]
        html_object = eli5.show_prediction(model,X[0],feature_names=feature_names,target_names = class_names)
        
        raw_html = html_object._repr_html_()
        components.html(raw_html, width=1100, height=350, scrolling=True)

    st.write("$Citation Request: Kok, I., Uzunkaya, M., Ozdemir, S., Bulbul, I. A., & Ozdemir, S. (2022). XAI-based Early Autism Diagnostic Interface.$ [Click for Document](https://aiot.cs.hacettepe.edu.tr/etjasd/XAI_Based_Early_Autism_Diognostic_Interface.pdf)")

    

#model 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

import warnings
warnings.filterwarnings('ignore')

#Reading data 
df=pd.read_csv("data/neo.csv")

#Label encoding output variable
label_encoder_hazardous = LabelEncoder()
df['hazardous'] = label_encoder_hazardous.fit_transform(df['hazardous'])

#Dividing data source to input/output variables for Creating Training & Testing Data set
#X = df.drop(['hazardous','id','name','orbiting_body','sentry_object'],axis=1)
#y = df.drop(['id','name','orbiting_body','sentry_object','est_diameter_min','est_diameter_max','relative_velocity',\
             #'miss_distance','absolute_magnitude'],axis=1)

X = df[['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance','absolute_magnitude']]
y = df[['hazardous']]

## over sampling
random_over_sampler = RandomOverSampler()
X, y = random_over_sampler.fit_resample(X, y)

#Scaling input variables
sc=StandardScaler()
X_scaled=pd.DataFrame(sc.fit_transform(X))
X_scaled.columns=X.columns

#Train & Test 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 0)

#model building
knn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 1)
knn.fit(X_train, y_train)
knn_Pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, knn_Pred)
print('Accuracy:', accuracy)

# save the model to disk
joblib.dump(knn, "knn_model.sav")
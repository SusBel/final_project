import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

# 1. טעינת ה-Golden Set
csv_path = r"C:\Finals_Project\GoldenSet\golden_set_logic.csv"
df = pd.read_csv(csv_path)

# 2. הגדרת המאפיינים (X) והתווית (y)
X = df[['input_intent', 'input_emotion', 'history_state']]
y = df['expected_response']

# 3. קידוד הטקסט לוקטורים מספריים (One-Hot Encoding)
# handle_unknown='ignore' אומר שאם בעתיד יגיע רגש שלא היה באימון, הוא לא יקרוס
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# 4. אימון מודל KNN
# נבחר K=3 (3 השכנים הקרובים ביותר) 
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_encoded, y)

# 5. שמירת המודל והמקודד לקבצים כדי שנוכל לטעון אותם ב-inference.py
with open('logic_knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
    
with open('logic_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("KNN Model and Encoder saved successfully!")
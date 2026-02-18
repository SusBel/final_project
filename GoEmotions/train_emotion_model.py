import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import os

# ==========================================
# 1. הגדרות (Configuration)
# ==========================================
TRAIN_FILE = 'GoEmotions/processed_emotions_train_BALANCED.csv'
DEV_FILE   = 'GoEmotions/processed_emotions_dev.csv'
TEST_FILE  = 'GoEmotions/processed_emotions_test.csv'

# לאן לשמור את המודל המוכן
MODEL_SAVE_PATH = 'emotion_model.keras'
LE_SAVE_PATH    = 'emotion_label_encoder.pkl'

# פרמטרים לאימון (Hyperparameters)
VOCAB_SIZE = 10000   # כמה מילים המודל יכיר? (10,000 הנפוצות ביותר)
MAX_LENGTH = 50      # אורך מקסימלי למשפט
EMBEDDING_DIM = 64   # גודל הוקטור לכל מילה (עומק ההבנה הסמנטית)
EPOCHS = 20          # מספר סיבובים מקסימלי על החומר

# ==========================================
# 2. טעינת והכנת הנתונים
# ==========================================
def load_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    # מוודאים שהטקסט הוא מסוג string
    return df['cleaned_text'].astype(str).values, df['emotion'].values

# טעינת שלושת הקבצים שהכנו
X_train, y_train = load_data(TRAIN_FILE)
X_val, y_val     = load_data(DEV_FILE)
X_test, y_test   = load_data(TEST_FILE)

# המרת התגיות ממילים למספרים (למשל: "anger" -> 0)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# שמירת המקרא (כדי שנוכל לתרגם חזרה את התשובות של המודל)
with open(LE_SAVE_PATH, 'wb') as f:
    pickle.dump(le, f)

# המרה ל"ווקטורים חמים" (One-Hot Encoding)
# המודל צריך פלט כזה: [0, 0, 1, 0, 0, 0] עבור רגש מספר 3
num_classes = len(le.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val_enc, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test_enc, num_classes)

# ==========================================
# 3. בניית המודל (The Architecture)
# ==========================================
# שכבת הוקטוריזציה: הופכת טקסט למספרים באופן אוטומטי
vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LENGTH
)
vectorizer.adapt(X_train) # המודל לומד את אוצר המילים מהאימון

print("Building Model...")
model = models.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string), # קלט: משפט טקסט רגיל
    vectorizer,                                  # תרגום: טקסט -> מספרים
    
    # שכבת Embedding: הופכת מספרים למשמעות (סמנטיקה)
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),
    
    # שכבת Pooling: מסכמת את כל המשפט לווקטור אחד שמייצג את "רוח הדברים"
    layers.GlobalAveragePooling1D(),
    
    # שכבות Dense: המוח שמקבל החלטות
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # מונע שינון (Overfitting) - כנדרש בפרויקט
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    # שכבת היציאה: הסתברות לכל אחד מ-7 הרגשות
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',                 # האלגוריתם שביקשת באפיון
    loss='categorical_crossentropy',  # פונקציית הטעות לסיווג רב-מחלקתי
    metrics=['accuracy']
)

# ==========================================
# 4. אימון (Training)
# ==========================================
# מנגנוני הגנה (Callbacks) כפי שהוגדר באפיון
callbacks_list = [
    # עצירה מוקדמת אם אין שיפור (מונע ביזבוז זמן ושינון יתר)
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    # שמירת המודל הטוב ביותר בלבד
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
]

print("Starting Training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

# ==========================================
# 5. בדיקה (Evaluation)
# ==========================================
print("\n--- Final Test Report ---")
# בדיקת דיוק כללי
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Total Accuracy: {accuracy:.2%}")

# דוח מפורט לפי רגשות (F1 Score)
y_pred_probs = model.predict(X_test)
y_pred_indices = np.argmax(y_pred_probs, axis=1)

# המרה חזרה ממספרים לשמות (0 -> "anger")
y_test_labels = le.inverse_transform(y_test_enc)
y_pred_labels = le.inverse_transform(y_pred_indices)

print("\nDetailed Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))
print(f"Model saved successfully to: {MODEL_SAVE_PATH}")
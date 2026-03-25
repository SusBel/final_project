import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# ==========================================
# 1. הגדרות
# ==========================================
TRAIN_FILE = 'GoEmotions/processed_emotions_train_BALANCED.csv'
DEV_FILE   = 'GoEmotions/processed_emotions_dev.csv'
TEST_FILE  = 'GoEmotions/processed_emotions_test.csv'

MODEL_SAVE_PATH = 'emotion_model_optimized.keras'
LE_SAVE_PATH    = 'emotion_label_encoder.pkl'

# ---> שינוי קריטי 1: חיתוך אוצר המילים ל-5000 למניעת שינון <---
VOCAB_SIZE  = 5000
EPOCHS      = 60
BATCH_SIZE  = 128   # באטצ' גדול עוזר ליציבות הלמידה

# ==========================================
# 2. טעינת הנתונים
# ==========================================
def load_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    return df['cleaned_text'].astype(str).values, df['emotion'].values

X_train, y_train = load_data(TRAIN_FILE)
X_val,   y_val   = load_data(DEV_FILE)
X_test,  y_test  = load_data(TEST_FILE)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

with open(LE_SAVE_PATH, 'wb') as f:
    pickle.dump(le, f)

num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")

y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val_enc,   num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test_enc,  num_classes)

# ==========================================
# 3. ווקטורייזר יחיד וממוקד (TF-IDF)
# ==========================================
# ---> שינוי קריטי 2: הורדת הכפילות. נשארנו רק עם ה-TF-IDF <---
tfidf_vec = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='tf_idf',
    name='tfidf_vec'
)
print("Adapting vocabulary...")
tfidf_vec.adapt(X_train)

# ==========================================
# 4. המודל — קטן, רזה ומוסדר
# ==========================================
print("\nBuilding Optimized Model...")

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")

# המרת טקסט למספרים
x = tfidf_vec(text_input)

# שכבה ראשונה ואחת בלבד! עם קנס (L2) כבד על משקולות גדולות
x = layers.Dense(64, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01))(x)

# Dropout חזק למניעת התאמת יתר (Overfitting)
x = layers.Dropout(0.5)(x)

# פלט
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(text_input, outputs)

# הדפסת כמות הפרמטרים - תראה שזה ירד משמעותית!
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,} (Much healthier for this dataset!)")

# ==========================================
# 5. קומפילציה
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # label_smoothing אומר למודל "אל תהיה בטוח בעצמך ב-100%, תמיד תשאיר מקום לספק" - מעולה נגד שינון
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==========================================
# 6. Callbacks (מנגנוני הגנה)
# ==========================================
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )
]

# ==========================================
# 7. אימון
# ==========================================
print("\nStarting Training...\n")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

best_val = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val) + 1
print(f"\nBest val_accuracy: {best_val:.2%} at epoch {best_epoch}")

# ==========================================
# 8. הערכה
# ==========================================
print("\n--- Final Test Report ---")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Total Accuracy on Test Set: {accuracy:.2%}")

y_pred_probs   = model.predict(X_test, verbose=0)
y_pred_indices = np.argmax(y_pred_probs, axis=1)
y_test_labels  = le.inverse_transform(y_test_enc)
y_pred_labels  = le.inverse_transform(y_pred_indices)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Label encoder saved to: {LE_SAVE_PATH}")
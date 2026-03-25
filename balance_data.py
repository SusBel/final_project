import pandas as pd
from nltk.corpus import wordnet
import random
import os


# ==========================================
# 1. פונקציית שינוי טקסט (Augmentation)
# ==========================================
def synonym_replacement(text, n=1):
    """
    מקבלת משפט ומחליפה n מילים במילים נרדפות רנדומליות.
    """
    words = text.split()
    if len(words) < 2:
        return text # משפט קצר מדי
        
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = []
        # חיפוש מילים נרדפות ב-WordNet
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' abcdefghijklmnopqrstuvwxyz'])
                if synonym != random_word and synonym not in synonyms:
                    synonyms.append(synonym)
        
        # אם מצאנו מילה נרדפת - נחליף
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
            
        if num_replaced >= n: # מספיק החלפות למשפט אחד
            break

    return ' '.join(new_words)

# ==========================================
# 2. הלוגיקה הראשית
# ==========================================
def smart_balance_dataset(df, text_col, label_col, target_count=3000):
    print(f"--- Smart Balancing (Target: {target_count}) ---")
    labels = df[label_col].unique()
    balanced_dfs = []
    
    for label in labels:
        label_df = df[df[label_col] == label]
        current_count = len(label_df)
        
        if current_count == 0: continue
            
        # 1. אם יש יותר מדי - חותכים (Downsample)
        if current_count > target_count:
            resampled = label_df.sample(n=target_count, random_state=42)
            print(f"[DOWN] '{label}': {current_count} -> {target_count} (Random Cut)")
            balanced_dfs.append(resampled)
            
        # 2. אם יש פחות מדי - משכפלים עם שינויים (Augmentation)
        elif current_count < target_count:
            # קודם כל לוקחים את כל מה שיש
            balanced_dfs.append(label_df)  
            
            # מחשבים כמה חסר
            missing = target_count - current_count
            print(f"[AUGMENT] '{label}': Generating {missing} new sentences...", end='\r')
            
            new_samples = []
            source_texts = label_df[text_col].tolist()
            
            # לולאה ליצירת משפטים חדשים
            for _ in range(missing):
                # בוחרים משפט קיים באקראי
                original_text = random.choice(source_texts)
                # יוצרים וריאציה שלו
                augmented_text = synonym_replacement(original_text)
                
                new_samples.append({
                    text_col: augmented_text,
                    label_col: label
                })
            
            # הופכים את החדשים ל-DataFrame ומוסיפים
            new_df = pd.DataFrame(new_samples)
            balanced_dfs.append(new_df)
            print(f"[AUGMENT] '{label}': {current_count} -> {target_count} (Added {missing} variations)")
            
        else:
            balanced_dfs.append(label_df)
            print(f"[KEEP] '{label}': Exact match")

    # איחוד וערבוב סופי
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df

# ==========================================
# 3. הרצה
# ==========================================
if __name__ == "__main__":
    # התקנת NLTK אוטומטית אם חסר (בתוך הקוד)
    try:
        wordnet.ensure_loaded()
    except:
        pass

    # --- טיפול ברגשות ---
    emo_path = 'GoEmotions/processed_emotions_train.csv'
    if os.path.exists(emo_path):
        print("\nProcessing Emotions...")
        df = pd.read_csv(emo_path)
        df_balanced = smart_balance_dataset(df, 'cleaned_text', 'emotion', target_count=3000) 
        df_balanced.to_csv('GoEmotions/processed_emotions_train_AUGMENTED.csv', index=False)
        print("Done! Saved to: processed_emotions_train_AUGMENTED.csv")

    # --- טיפול בכוונות ---
    intent_path = 'processed_intents.csv'
    if os.path.exists(intent_path):
        print("\nProcessing Intents...")
        df = pd.read_csv(intent_path)
        df_balanced = smart_balance_dataset(df, 'text', 'intent', target_count=3000)
        df_balanced.to_csv('processed_intents_AUGMENTED.csv', index=False)
        print("Done! Saved to: processed_intents_AUGMENTED.csv")
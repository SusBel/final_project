import pandas as pd
import json
import re
import os

# ==========================================
# 1. Configuration & Setup
# ==========================================

# Load the Custom Mapping from the JSON file
try:
    with open('GoEmotions/emotion_mapping.json', 'r') as f:
        PROJECT_EMOTION_MAPPING = json.load(f)
    print("Loaded mapping from emotion_mapping.json")
except FileNotFoundError:
    print("CRITICAL ERROR: Could not find 'emotion_mapping.json'.")
    print("Please create this file with your emotion categories.")
    exit()

# Load the list of 28 raw emotion names (indices 0-27)
# This file comes with the GoEmotions dataset
try:
    with open('GoEmotions/emotions.txt', 'r') as f:
        emotion_labels = f.read().splitlines()
        id2label = {i: label for i, label in enumerate(emotion_labels)}
except FileNotFoundError:
    print("Error: Could not find 'GoEmotions/emotions.txt'. Please check the path.")
    exit()

# ==========================================
# 2. Helper Functions
# ==========================================

def clean_text(text):
    """Basic text cleaning: lowercase, remove special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s?!.,]', '', text) # Keep punctuation for context
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(file_path):
    """Reads a TSV file, maps labels using loaded JSON, and returns a DataFrame."""
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (File not found)")
        return None

    print(f"Processing {file_path}...")
    
    # Read TSV (The files have no headers: text, label_ids, id)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label_ids', 'id'])
    
    # 1. Clean Text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 2. Map Labels Logic
    def get_project_emotion(label_str):
        if pd.isna(label_str):
            return 'neutral'
        
        # Convert "27,4" (string) -> [27, 4] (integers)
        ids = [int(x) for x in label_str.split(',')]
        
        # Get original names: e.g., ["neutral", "sadness"]
        original_names = [id2label[i] for i in ids if i in id2label]
        
        # Map to Project Categories using the loaded JSON
        mapped_labels = [PROJECT_EMOTION_MAPPING.get(name, 'neutral') for name in original_names]
        
        # Priority Logic: Return the first non-neutral emotion if it exists
        # e.g., if input is "curiosity (neutral), anger" -> we want "anger"
        non_neutral = [label for label in mapped_labels if label != 'neutral']
        
        if non_neutral:
            return non_neutral[0] # Return the first strong emotion found
        
        return 'neutral' # Default if only neutral found

    df['emotion'] = df['label_ids'].apply(get_project_emotion)
    
    # Return only necessary columns
    return df[['cleaned_text', 'emotion']]

# ==========================================
# 3. Execution
# ==========================================

# Define input files
files_to_process = {
    'train': 'GoEmotions/train.tsv',
    'dev':   'GoEmotions/dev.tsv',
    'test':  'GoEmotions/test.tsv'
}

# Process loop
for split_name, file_path in files_to_process.items():
    df = process_file(file_path)
    
    if df is not None:
        # Save output
        output_name = f'GoEmotions/processed_emotions_{split_name}.csv'
        df.to_csv(output_name, index=False)
        print(f"Saved {output_name}")
        
        # Print distribution for the training set only (to check balance)
        if split_name == 'train':
            print("\nTrain Set Distribution:")
            print(df['emotion'].value_counts())

print("\nData preparation complete.")
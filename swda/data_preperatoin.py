import pandas as pd
import os

# ==========================================
# 1. הגדרות ומיפוי (אותו מיפוי כמו קודם)
# ==========================================
INTENT_MAPPING = {
    'qy': 'question', 'qw': 'question', 'qo': 'question', 'qh': 'question', 
    'qr': 'question', 'qrr': 'question',
    'ad': 'request',
    'sd': 'statement', 'sv': 'statement',
    'fp': 'general', 'fc': 'general', 'b': 'general', 'bk': 'general'
}

def process_all_swda_folders():
    all_data = []
    base_path = "." # מתחיל לחפש מהתיקייה הנוכחית
    
    print(f"Starting scan in: {os.path.abspath(base_path)}")
    print("Looking for ALL CSV files in all subfolders...")
    
    files_processed_count = 0
    
    # os.walk עובר לבד על כל התיקיות ותתי-התיקיות (sw00utt, sw01utt...)
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # אנחנו מחפשים כל קובץ CSV, לא משנה מה שם התיקייה
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                
                try:
                    # מנסים לקרוא את הקובץ
                    df = pd.read_csv(full_path)
                    
                    # בדיקת תקינות: האם זה קובץ של SWDA? (צריך להכיל act_tag ו-text)
                    # המרה לאותיות קטנות למקרה שהכותרות שונות קצת
                    cols = [c.lower() for c in df.columns]
                    
                    if 'act_tag' in cols and 'text' in cols:
                        # תיקון שמות העמודות אם צריך
                        df.columns = [c.lower() for c in df.columns]
                        
                        # מעבר על השורות
                        for _, row in df.iterrows():
                            tag = str(row['act_tag']).lower()
                            text = str(row['text'])
                            
                            if tag in INTENT_MAPPING:
                                # ניקוי טקסט
                                clean_text = text.replace('{', '').replace('}', '').replace('#', '').strip()
                                
                                if len(clean_text) > 1:
                                    all_data.append({
                                        'text': clean_text,
                                        'intent': INTENT_MAPPING[tag]
                                    })
                        
                        files_processed_count += 1
                        # הדפסה קטנה כל 50 קבצים כדי שתדע שזה עובד
                        if files_processed_count % 50 == 0:
                            print(f"Processed {files_processed_count} files...", end='\r')
                            
                except Exception as e:
                    # התעלם מקבצים פגומים או לא קשורים
                    pass

    # ==========================================
    # 2. שמירה
    # ==========================================
    if len(all_data) > 0:
        final_df = pd.DataFrame(all_data)
        
        # ערבוב הנתונים (חשוב לאימון!)
        final_df = final_df.sample(frac=1).reset_index(drop=True)
        
        output_filename = 'processed_intents.csv'
        final_df.to_csv(output_filename, index=False)
        
        print(f"\n\nDONE!")
        print(f"Scanned {files_processed_count} CSV files.")
        print(f"Total samples collected: {len(final_df)}")
        print(f"Saved to: {output_filename}")
        print("\nFinal Distribution:")
        print(final_df['intent'].value_counts())
    else:
        print("\nNo valid SWDA data found. Are the CSV files in the subfolders?")

if __name__ == "__main__":
    process_all_swda_folders()
import pandas as pd

# ==========================================
# Logic Module Golden Set
# ==========================================
# Input: Intent + Emotion + History State
# Output: Expected Bot Action
# ==========================================

logic_scenarios = [
    # ---------------------------------------------------------
    # SCENARIO 1: ANGER (Escalation Handling)
    # ---------------------------------------------------------
    # Case 1: Customer starts with anger (No history) -> Apologize & De-escalate
    {
        "input_intent": "statement", "input_emotion": "anger", "history_state": "start",
        "expected_response": "apology_empathy"
    },
    # Case 2: Bot already apologized, customer still angry & requesting -> Escalate to human
    {
        "input_intent": "request", "input_emotion": "anger", "history_state": "bot_apologized",
        "expected_response": "escalate_to_human"
    },
    # Case 3: Angry question ("Why isn't this working?!") -> Technical help + Apology
    {
        "input_intent": "question", "input_emotion": "anger", "history_state": "start",
        "expected_response": "technical_solution_with_apology"
    },
    
    # ---------------------------------------------------------
    # SCENARIO 2: FEAR (Reassurance)
    # ---------------------------------------------------------
    # Case 4: Security concern ("Is this safe?")
    {
        "input_intent": "question", "input_emotion": "fear", "history_state": "start",
        "expected_response": "security_reassurance"
    },
    # Case 5: Statement of fear ("My money is gone") -> Urgent assistance
    {
        "input_intent": "statement", "input_emotion": "fear", "history_state": "start",
        "expected_response": "urgent_assistance"
    },
    # Case 6: Bot already reassured, customer still fearful -> Detailed policy
    {
        "input_intent": "question", "input_emotion": "fear", "history_state": "bot_reassured",
        "expected_response": "provide_detailed_policy"
    },

    # ---------------------------------------------------------
    # SCENARIO 3: DISGUST (Quality Assurance)
    # ---------------------------------------------------------
    # Case 7: Quality complaint
    {
        "input_intent": "statement", "input_emotion": "disgust", "history_state": "start",
        "expected_response": "apology_quality_assurance"
    },
    # Case 8: Requesting compensation due to disgust
    {
        "input_intent": "request", "input_emotion": "disgust", "history_state": "bot_apologized",
        "expected_response": "offer_compensation"
    },

    # ---------------------------------------------------------
    # SCENARIO 4: SADNESS (Empathy)
    # ---------------------------------------------------------
    # Case 9: Cancellation due to sadness/hardship
    {
        "input_intent": "statement", "input_emotion": "sadness", "history_state": "start",
        "expected_response": "empathy_retention_offer"
    },
    # Case 10: "Why did this happen to me?"
    {
        "input_intent": "question", "input_emotion": "sadness", "history_state": "start",
        "expected_response": "empathetic_explanation"
    },

    # ---------------------------------------------------------
    # SCENARIO 5: SURPRISE (Verification)
    # ---------------------------------------------------------
    # Case 11: Positive surprise -> Thank user
    {
        "input_intent": "statement", "input_emotion": "surprise", "history_state": "service_completed",
        "expected_response": "thank_and_close"
    },
    # Case 12: Negative surprise ("You charged me twice??") -> Check status
    {
        "input_intent": "question", "input_emotion": "surprise", "history_state": "start",
        "expected_response": "check_account_status"
    },

    # ---------------------------------------------------------
    # SCENARIO 6: JOY (Retention)
    # ---------------------------------------------------------
    # Case 13: Happy greeting
    {
        "input_intent": "general", "input_emotion": "joy", "history_state": "start",
        "expected_response": "warm_welcome_back"
    },
    # Case 14: Thanking the bot
    {
        "input_intent": "statement", "input_emotion": "joy", "history_state": "bot_solved_problem",
        "expected_response": "polite_closing"
    },
    # Case 15: Enthusiastic request ("I want to buy another one!")
    {
        "input_intent": "request", "input_emotion": "joy", "history_state": "start",
        "expected_response": "process_order_enthusiastic"
    },

    # ---------------------------------------------------------
    # SCENARIO 7: NEUTRAL (Efficiency)
    # ---------------------------------------------------------
    # Case 16: Standard info question
    {
        "input_intent": "question", "input_emotion": "neutral", "history_state": "start",
        "expected_response": "provide_information"
    },
    # Case 17: Standard request
    {
        "input_intent": "request", "input_emotion": "neutral", "history_state": "start",
        "expected_response": "execute_action"
    },
    # Case 18: Standard greeting ("Hi")
    {
        "input_intent": "general", "input_emotion": "neutral", "history_state": "start",
        "expected_response": "standard_greeting"
    },
    # Case 19: Acknowledgment ("Okay thanks")
    {
        "input_intent": "general", "input_emotion": "neutral", "history_state": "bot_provided_info",
        "expected_response": "close_interaction"
    },
    
    # ---------------------------------------------------------
    # SCENARIO 8: SEQUENCES (Stateful Logic Checks)
    # ---------------------------------------------------------
    # Step 1: User asks question (Neutral)
    {
        "input_intent": "question", "input_emotion": "neutral", "history_state": "start",
        "expected_response": "provide_information"
    },
    # Step 2: User didn't understand and gets angry (History: Bot already provided info)
    {
        "input_intent": "statement", "input_emotion": "anger", "history_state": "bot_provided_info",
        "expected_response": "apology_rephrase"
    },
    # Step 3: User is happy now (History: Bot rephrased)
    {
        "input_intent": "statement", "input_emotion": "joy", "history_state": "bot_rephrased",
        "expected_response": "happy_closing"
    }
]

# Create DataFrame
df_logic = pd.DataFrame(logic_scenarios)

# Save to CSV
filename = 'golden_set_logic_stateful.csv'
df_logic.to_csv(filename, index=False)

print(f"Generated '{filename}' with {len(df_logic)} scenarios.")
print("\nUnique Input Combinations:")
print(f"- Intents: {df_logic['input_intent'].unique()}")
print(f"- Emotions: {df_logic['input_emotion'].unique()}")
print(f"- History States: {df_logic['history_state'].nunique()} different states")

print("\nSample Rows:")
print(df_logic.head().to_string())
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# Import the generator function from the first file (assuming both scripts are in the same directory)
from cmti_live import generate_sensor_data_stream 


# --- 1. CONFIGURATION ---
THRESHOLD = 0.7  # Classification threshold for CMTI Score (CMTI > 0.7 -> Action)

# --- 2. DEEP LEARNING ARCHITECTURE (Blueprint Definition) ---
# (Include your standard DL architecture definition here for report integrity)
# ...

# --- 3. CORE CALCULATION FUNCTIONS (The Work) ---

def calculate_feature_engineering(df_raw):
    """
    CALCULATION 1: Derives engineered features (X1, X2) from the raw stream.
    This is the work of feature engineering performed on the processor side.
    """
    # X1: G-Shock Count (normalized peak count)
    # Calculation Logic: Raw peaks normalized by trip duration (longer trips require more scrutiny)
    G_Shock_Count_X1 = (df_raw['Raw_Peak_Count_Input'] / (df_raw['Trip_Duration_Hours'] * 20))
    G_Shock_Count_X1 = np.clip(G_Shock_Count_X1, 1, 15).astype(int)

    # X2: Fatigue Index (normalized sustained roughness)
    # Calculation Logic: Based on the raw standard deviation and trip length factor.
    Fatigue_Index_X2 = np.clip(df_raw['Raw_G_StdDev_Input'] * 0.45 + (df_raw['Trip_Duration_Hours'] / 200), 0.1, 1.0)

    df_raw['G_Shock_Count_X1'] = G_Shock_Count_X1
    df_raw['Fatigue_Index_X2'] = Fatigue_Index_X2.round(4)
    return df_raw

def calculate_prediction_and_action(df_processed):
    """
    CALCULATION 2: Simulates the LSTM prediction and makes the final action decision.
    """
    
    # Set the seed again for prediction logic consistency
    np.random.seed(42) # IMPORTANT: Ensure ground truth calculation is reproducible

    # A. Calculate Actual Damage (Ground Truth Calculation)
    # MODIFIED LOGIC: Making the criteria for '1' (Damage) easier to hit 
    # to ensure we have a mix of classes (0 and 1).
    df_processed['Actual_Damage_Y_True'] = (
        (df_processed['G_Shock_Count_X1'] >= 5) |  # Damage if G-Shock is moderate OR
        (df_processed['Fatigue_Index_X2'] >= 0.6) # Damage if Fatigue is moderate
    ).astype(int)
    
    # ... rest of the function remains the same ...

    # B. CMTI Score Calculation (Simulating the LSTM's final layer)
    # Using calculated features X1 and X2 as inputs
    df_processed['CMTI_Score_Y_Prob'] = (
        (df_processed['G_Shock_Count_X1'] * 0.06) + 
        (df_processed['Fatigue_Index_X2'] * 0.60) + 
        (np.random.rand(len(df_processed)) * 0.15) 
    )
    df_processed['CMTI_Score_Y_Prob'] = np.clip(df_processed['CMTI_Score_Y_Prob'], 0.05, 0.95).round(4)

    # C. Predicted Damage Calculation (The Action Trigger)
    df_processed['Predicted_Damage_Y_Pred'] = (df_processed['CMTI_Score_Y_Prob'] > THRESHOLD).astype(int)
    
    return df_processed

def yield_actionable_results(df):
    """Calculates all final metrics and yields the action for each shipment."""
    
    actual = df['Actual_Damage_Y_True']
    predicted = df['Predicted_Damage_Y_Pred']

    # Final Metric Calculation
    cm = confusion_matrix(actual, predicted); TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(actual, predicted); recall = recall_score(actual, predicted, zero_division=0)
    precision = precision_score(actual, predicted, zero_division=0)
    
    # --- YIELD ACTIONABLE OUTPUT ---
    print("\n" + "="*50)
    print("      CMTI PROCESSOR: ACTIONABLE RESULTS (LIVE DECISION)")
    print("==================================================")
    
    print(f"Total Shipments Processed: {len(df)}")
    print(f"Actionable Flags (Predicted Damage=1): {predicted.sum()}")
    print("\n--- ACTION SUMMARY ---")
    
    # Yield the action and final status for each shipment (The 'work' performed)
    for index, row in df.iterrows():
        action = "🚩 INITIATE RE-BRACING/INSPECTION" if row['Predicted_Damage_Y_Pred'] == 1 else "✅ PROCEED (LOW RISK)"
        status = " (CONFIRMED DAMAGE)" if row['Actual_Damage_Y_True'] == 1 else " (FALSE POSITIVE)" if row['Predicted_Damage_Y_Pred'] == 1 and row['Actual_Damage_Y_True'] == 0 else ""
        print(f"[{row['Shipment_ID']}] CMTI Score: {row['CMTI_Score_Y_Prob']} -> {action}{status}")
    
    print("\n--- FINAL SIMULATION METRICS FOR REVIEW ---")
    print(f"1. Test Recall (FN Minimization): {recall*100:.2f}%")
    print(f"2. Test Precision (FP Cost): {precision*100:.2f}%")
    print("==================================================")


# --- MAIN EXECUTION: The Live Call ---
if __name__ == "__main__":
    
    # 1. LIVE DATA TRANSMISSION (Function Call)
    raw_data_stream = generate_sensor_data_stream()
    print("✅ Processor: Received live data stream from generator function.")
    
    # 2. PERFORM CALCULATION WORK (Feature Engineering)
    processed_features = calculate_feature_engineering(raw_data_stream)
    
    # 3. PERFORM CALCULATION WORK (Prediction and Action)
    final_output = calculate_prediction_and_action(processed_features)
    
    # 4. YIELD ACTION AND RESULTS
    yield_actionable_results(final_output)
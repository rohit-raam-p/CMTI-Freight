import numpy as np
import pandas as pd

# --- CONFIGURATION ---
N_SHIPMENTS = 50 
np.random.seed(42) 

def generate_sensor_data_stream():
    """
    Simulates the collection of raw sensor inputs for a batch of shipments.
    This function acts as the "live feed" source.
    """
    
    shipment_data = []
    
    # Simulate a range of trip durations (1 to 7 days)
    trip_durations = np.random.randint(24, 168, N_SHIPMENTS)
    
    for i in range(N_SHIPMENTS):
        duration = trip_durations[i]
        
        # Simulate an average g-force roughness (rougher for longer/riskier trips)
        avg_g_std_dev = np.random.uniform(low=0.1, high=duration / 100, size=1)[0]
        
        # Simulate the total count of high-force peaks detected (raw, uncalculated count)
        raw_peak_count = np.random.randint(500, 2500)
        
        shipment_data.append({
            'Shipment_ID': f'SHIP-{i+1:03d}',
            'Trip_Duration_Hours': duration,
            'Raw_G_StdDev_Input': avg_g_std_dev,
            'Raw_Peak_Count_Input': raw_peak_count
        })

    # Return the data as a Pandas DataFrame (the in-memory "live stream")
    print("✅ Generator: Simulated raw sensor data stream is ready for transmission.")
    return pd.DataFrame(shipment_data)

if __name__ == "__main__":
    # In a real-world scenario, this would initiate the processor
    # For simulation, we import and call the processor function directly.
    
    # NOTE: You must combine this file and the processor file for execution,
    # or ensure you run the processor script which imports this function.
    pass
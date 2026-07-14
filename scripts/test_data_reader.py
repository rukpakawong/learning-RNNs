import torch
import sys

# Import the generic function we just wrote
try:
    from data_reader import load_and_prepare_time_series_data
except ImportError:
    print("Error: Could not import 'data_reader.py'. Make sure it is in the same folder.")
    sys.exit(1)

def run_covid_test():
    print("Starting integration test with GCP COVID-19 Open Data...\n")
    
    # Official GCP COVID-19 Open Data URL for the US
    covid_url = "https://storage.googleapis.com/covid19-open-data/v3/location/US.csv"
    
    try:
        # We tell our generic reader exactly how to handle this specific CSV
        train_loader, val_loader, val_dataset, scaler = load_and_prepare_time_series_data(
            filepath_or_url=covid_url,
            target_column='new_confirmed', # The column we want to predict
            date_column='date',            # The column to sort by
            seq_length=14,                 # Look at the past 14 days
            batch_size=32                  # 32 sequences per batch
        )
        
        print("\n--- Tensor Shape Verification ---")
        
        # Fetch exactly one batch from the train_loader to inspect its shape
        for batch_x, batch_y in train_loader:
            print(f"Input X shape: {batch_x.shape}")
            print(f" -> [Batch Size: {batch_x.shape[0]}, Sequence Length: {batch_x.shape[1]}, Features: {batch_x.shape[2] if len(batch_x.shape) > 2 else 1}]")
            
            print(f"\nTarget Y shape: {batch_y.shape}")
            print(f" -> [Batch Size: {batch_y.shape[0]}, Features: {batch_y.shape[1] if len(batch_y.shape) > 1 else 1}]")
            
            # Verify data types are correct for PyTorch neural networks
            assert isinstance(batch_x, torch.Tensor), "Input X is not a PyTorch Tensor!"
            assert isinstance(batch_y, torch.Tensor), "Target Y is not a PyTorch Tensor!"
            assert batch_x.dtype == torch.float32, "Input X must be float32!"
            
            print("\nSUCCESS: The generic data reader correctly downloaded, parsed, scaled, and batched the live COVID-19 data!")
            break # We only need to check the first batch

    except Exception as e:
        print(f"\nTEST FAILED: An error occurred during the data loading process.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    run_covid_test()
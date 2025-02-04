import cupy as cp
import numpy as np
import pandas as pd
import time
import os
import logging
from decimal import Decimal, getcontext

# Set up logging
os.makedirs("data/analysis", exist_ok=True)
logging.basicConfig(filename="data/analysis/collatz_analysis.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set Decimal precision
getcontext().prec = 100000  # High precision for large numbers

def collatz_sequence(n):
    sequence = []
    while n != 1:
        sequence.append(n)
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
    sequence.append(1)
    return sequence

# Function to compute Collatz steps using CuPy
def collatz_steps(n):
    steps = 0
    odd_steps = 0
    even_steps = 0
    odd_descents = 0
    log2_values = []
    
    for n in collatz_sequence(n):
        steps += 1
        if n % 2 == 0:
            even_steps += 1
        else:
            odd_steps += 1
            if odd_steps > 1 and n < prev_n:
                odd_descents += 1
        log2_values.append(np.log2(n))
        prev_n = n

    return steps, odd_steps, even_steps, odd_descents, log2_values

# GPU-optimized computation for multiple numbers
def generate_collatz_data(start, end):
    numbers = cp.arange(start, end + 1)  # Create GPU array
    steps = cp.zeros(len(numbers), dtype=int)  # Initialize step counts
    odd_steps = cp.zeros(len(numbers), dtype=int)
    even_steps = cp.zeros(len(numbers), dtype=int)
    odd_descents = cp.zeros(len(numbers), dtype=int)
    log2_values_list = []

    for i in range(len(numbers)):
        steps[i], odd_steps[i], even_steps[i], odd_descents[i], log2_values = collatz_steps(numbers[i].item())
        log2_values_list.append(log2_values)  # Convert log2 values to float for storage
    
    return numbers.get(), steps.get(), odd_steps.get(), even_steps.get(), odd_descents.get(), [list(map(float, logs)) for logs in log2_values_list]

# Analyze and save data in chunks of 10000 numbers
def analyze_and_save(start, end, chunk_size=10000, output_dir="data/analysis"):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    for chunk_start in range(start, end + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end)
        numbers, steps, odd_steps, even_steps, odd_descents, log2_values_list = generate_collatz_data(chunk_start, chunk_end)
        
        df = pd.DataFrame({
            "Number": numbers, 
            "Steps": steps,
            "Odd Steps": odd_steps,
            "Even Steps": even_steps,
            "Odd Descents": odd_descents,
            "Log2 Values": [str(logs) for logs in log2_values_list], # Convert list to string for saving
            "Collatz Sequence": [list(map(int, collatz_sequence(n))) for n in numbers]
        })
        
        # Statistical insights
        stats = {
            "Mean Steps": np.mean(steps),
            "Median Steps": np.median(steps),
            "Max Steps": np.max(steps),
            "Min Steps": np.min(steps),
            "Variance": np.var(steps),
            "Standard Deviation": np.std(steps),
            "Mean Odd Steps": np.mean(odd_steps),
            "Mean Even Steps": np.mean(even_steps),
            "Mean Odd Descents": np.mean(odd_descents)
        }
        
        # Save data
        output_file = os.path.join(output_dir, f"collatz_report_{chunk_start}_{chunk_end}.csv")
        df.to_csv(output_file, index=False)
        
        # Save metrics
        metrics_file = output_file.replace(".csv", "_metrics.txt")
        with open(metrics_file, "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Data saved to {output_file}, Metrics saved to {metrics_file}")
        print(f"Data saved to {output_file}, Metrics saved to {metrics_file}")
    
    end_time = time.time()
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    
    # Save metrics
    metrics_file = output_file.replace(".csv", "_metrics.txt")
    with open(metrics_file, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    end_time = time.time()
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    logging.info(f"Data saved to {output_file}, Metrics saved to {metrics_file}")
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    print(f"Data saved to {output_file}, Metrics saved to {metrics_file}")
    return stats

# Run analysis for large numbers
if __name__ == "__main__":
    start_number = int(Decimal("1"))  # Large number start
    end_number = int(Decimal("2000000"))    # Range of 100 numbers
    analyze_and_save(start_number, end_number)
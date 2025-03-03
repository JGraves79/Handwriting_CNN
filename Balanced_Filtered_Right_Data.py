import pandas as pd

# Paths to your filtered CSVs
filtered_right_csv = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Filtered_Right_Data.csv"
output_balanced_right_csv = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Balanced_Filtered_Right_Data.csv"

# Step 1: Read the right-hand CSV
right_df = pd.read_csv(filtered_right_csv)

# Step 2: Randomly sample exactly 46 WIDs from the right-hand data
balanced_right_df = right_df.sample(n=46, random_state=1)  # `random_state=1` ensures reproducibility

# Step 3: Save the balanced Right-Hand Data to a new CSV
balanced_right_df.to_csv(output_balanced_right_csv, index=False)

print(f"Balanced right-hand data saved to: {output_balanced_right_csv}")

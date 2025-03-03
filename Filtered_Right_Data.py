import pandas as pd

# Load the CSV file
input_file = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Source_Data\CSAFE_Handwriting_Info.csv"
handwriting_info_df = pd.read_csv(input_file)

# Filter for rows where 'hand' is "Right" or "right"
filtered_df = handwriting_info_df[handwriting_info_df['hand'].isin(["Right", "right"])]

# Save the filtered data to a new CSV file
output_file = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Filtered_Right_Data.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to: {output_file}")

import pandas as pd

# Load the CSV file
handwriting_info_df = pd.read_csv(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Source_Data\CSAFE_Handwriting_Info.csv')


# Get all unique values in the "hand" column
unique_values = handwriting_info_df['hand'].unique()

# Print all unique values
print(f"All unique values in the 'hand' column: {unique_values}")
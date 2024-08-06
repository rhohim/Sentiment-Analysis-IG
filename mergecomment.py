import pandas as pd
import glob

# Assuming all CSV files are in the same directory and follow a similar naming pattern, e.g., file_1.csv, file_2.csv, ..., file_27.csv
csv_files = glob.glob(r'D:\CrevHim\Code\software\Sentiment Analysis Instagram\sentiment\3 KATA HARI INI\week4.csv')

# Initialize an empty list to store the data
data_list = []

# Iterate through each CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract the "Comment Text" column and append each row to the data list
    data_list.extend(df['Comment Text'].tolist())

# Create a new DataFrame with the 'text' column
result_df = pd.DataFrame({'text': data_list})

# Display the resulting DataFrame
# print(result_df)

# Save the result DataFrame to a new CSV file
result_df.to_csv('result3katahariiniweek4', index=False)
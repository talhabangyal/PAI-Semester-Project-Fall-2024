# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # For checking file existence

# Step 1: Define the CryptoData class and organize data
class CryptoData:
    """
    A class to represent and process brand data for sentiment analysis.

    Attributes:
        data (pd.DataFrame): DataFrame containing the raw data
    Methods:
        clean_data(): Cleans and structures the dataset
        handle_missing_values(): Handles missing values in the dataset
        calculate_sentiment_rolling_averages(window): Calculates rolling averages
    """
    
    def __init__(self, file_path):
        """
        Initializes the CryptoData object and loads the dataset.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}. Please check the file path.")
        
        # Read the data from the file
        self.data = pd.read_excel(file_path)
    
    def clean_data(self):
        """
        Cleans and structures the dataset.
        """
        # Renaming columns for consistency
        self.data.columns = [col.strip().replace(" ", "_") for col in self.data.columns]
        
        # Drop rows with invalid ages (negative or missing)
        self.data = self.data[(self.data['Age'] >= 0) & (self.data['Age'].notna())]
        
        # Ensure Satisfaction_Percentage is within 0-100 range
        self.data['Satisfaction_Percentage'] = self.data['Satisfaction_Percentage'].clip(lower=0, upper=100)
        
    def handle_missing_values(self):
        """
        Handles missing values in the dataset.
        """
        # Fill missing Satisfaction_Percentage with the mean value
        self.data['Satisfaction_Percentage'].fillna(self.data['Satisfaction_Percentage'].mean(), inplace=True)
        
        # Fill missing Product_Reviews with 'Unknown'
        self.data['Product_Reviews'].fillna('Unknown', inplace=True)
        
        # Fill other missing values with 'Not Specified'
        self.data.fillna('Not Specified', inplace=True)
    
    def calculate_sentiment_rolling_averages(self, window=3):
        """
        Calculates rolling averages for Satisfaction_Percentage.
        """
        self.data['Sentiment_Rolling_Average'] = self.data['Satisfaction_Percentage'].rolling(window=window).mean()

# Step 2: Load and clean the data
file_path = "Product_reviews.xlsx"  # Replace with the correct path if necessary
try:
    crypto_data = CryptoData(file_path)
    crypto_data.clean_data()
    crypto_data.handle_missing_values()
except FileNotFoundError as e:
    print(e)
    exit()

# Step 3: Perform analysis
class SentimentAnalysis(CryptoData):
    """
    A subclass for performing sentiment analysis.

    Methods:
        calculate_statistics(): Calculates mean, median, std deviation
        find_most_positive_negative(): Finds posts with most positive/negative sentiments
        correlate_sentiments(): Correlates sentiment trends between brands
    """
    
    def calculate_statistics(self):
        """
        Calculates mean, median, and standard deviation of Satisfaction_Percentage.
        """
        stats = {
            'Mean': self.data['Satisfaction_Percentage'].mean(),
            'Median': self.data['Satisfaction_Percentage'].median(),
            'Standard_Deviation': self.data['Satisfaction_Percentage'].std()
        }
        return stats
    
    def find_most_positive_negative(self):
        """
        Finds the most positive and negative posts.
        """
        most_positive = self.data.loc[self.data['Satisfaction_Percentage'].idxmax()]
        most_negative = self.data.loc[self.data['Satisfaction_Percentage'].idxmin()]
        return most_positive, most_negative
    
    def correlate_sentiments(self):
        """
        Finds correlation between brands.
        """
        # Grouping by brand and calculating mean sentiment
        if 'Brand' in self.data.columns:
            brand_sentiments = self.data.groupby('Brand')['Satisfaction_Percentage'].mean()
            return brand_sentiments.corr()
        return None

# Instantiate the SentimentAnalysis class
sentiment_analysis = SentimentAnalysis(file_path)
sentiment_analysis.clean_data()
sentiment_analysis.handle_missing_values()

# Calculate rolling averages
sentiment_analysis.calculate_sentiment_rolling_averages()

# Perform analysis
stats = sentiment_analysis.calculate_statistics()
most_positive, most_negative = sentiment_analysis.find_most_positive_negative()
correlation = sentiment_analysis.correlate_sentiments()

# Print analysis results
print("Statistics:", stats)
print("Most Positive Review:", most_positive)
print("Most Negative Review:", most_negative)
print("Correlation of Sentiments:", correlation if correlation else "Correlation could not be calculated.")

# Step 4: Visualization
# Line chart for sentiment trends over time
plt.figure(figsize=(10, 6))
plt.plot(sentiment_analysis.data['Sentiment_Rolling_Average'], label='Sentiment Rolling Average', color='blue')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Index')
plt.ylabel('Satisfaction Percentage')
plt.legend()
plt.show()

# Bar chart for average sentiment per brand
plt.figure(figsize=(8, 6))
brand_avg = sentiment_analysis.data.groupby('Brand')['Satisfaction_Percentage'].mean()
brand_avg.plot(kind='bar', title='Average Sentiment Per Brand', color='skyblue')
plt.ylabel('Average Satisfaction Percentage')
plt.show()

# Pie chart for sentiment distribution
plt.figure(figsize=(8, 8))
sentiment_distribution = sentiment_analysis.data['Product_Reviews'].value_counts()
sentiment_distribution.plot(kind='pie', autopct='%1.1f%%', title='Sentiment Distribution')
plt.ylabel('')
plt.show()

# Step 5: Save results to CSV
output_file = "cleaned_transformed_data.csv"
sentiment_analysis.data.to_csv(output_file, index=False)

# Step 6: Summarize findings
summary = """
The analysis highlights the following:
1. The mean satisfaction percentage is {:.2f}.
2. The most positive review came from {} with a sentiment score of {:.2f}.
3. The most negative review came from {} with a sentiment score of {:.2f}.
""".format(
    stats['Mean'], most_positive['Name'], most_positive['Satisfaction_Percentage'],
    most_negative['Name'], most_negative['Satisfaction_Percentage']
)
print(summary)
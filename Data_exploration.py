import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save outputs if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset with the appropriate encoding and delimiter
data = pd.read_csv('Data_sample.csv', encoding='windows-1254', sep=';', encoding_errors='ignore')

# Print the column names to identify the correct ones
print("Column Names in Dataset:")
print(data.columns)

# Handle missing values by filling or dropping
# Drop columns with too many missing values (e.g., more than 50%)
threshold = len(data) * 0.5
data = data.dropna(thresh=threshold, axis=1)

# Fill missing values for remaining columns
if 's:description' in data.columns:
    data['s:description'].fillna('No description available', inplace=True)
if 's:brand' in data.columns:
    data['s:brand'].fillna('Unknown Brand', inplace=True)
if 's:category' in data.columns:
    data['s:category'].fillna('Uncategorized', inplace=True)
if 's:breadcrumb' in data.columns:
    data['s:breadcrumb'].fillna('No breadcrumb', inplace=True)

# Drop duplicates if any
data = data.drop_duplicates()

# Save the cleaned data to a CSV file
cleaned_data_path = os.path.join('cleaned_data.csv')
data.to_csv(cleaned_data_path, index=False)

print(f"Cleaned data saved to {cleaned_data_path}")

# Perform EDA and save plots as .png images
# 1. Distribution of Product Categories
if 's:category' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x='s:category', data=data, order=data['s:category'].value_counts().index[:10])
    plt.title('Top 10 Product Categories')
    plt.xticks(rotation=90)
    category_dist_path = os.path.join(output_dir, 'category_distribution.png')
    plt.savefig(category_dist_path)
    plt.show()
    print(f"Category distribution plot saved to {category_dist_path}")
    
    # Save category distribution data to CSV
    category_counts = data['s:category'].value_counts().head(10)
    category_counts.to_csv(os.path.join(output_dir, 'category_distribution.csv'), header=True)
    print(f"Category distribution data saved to CSV.")

# 2. Distribution of Description Lengths
if 's:description' in data.columns:
    data['Description_Length'] = data['s:description'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Description_Length'], kde=True)
    plt.title('Distribution of Product Description Lengths')
    description_length_dist_path = os.path.join(output_dir, 'description_length_distribution.png')
    plt.savefig(description_length_dist_path)
    plt.show()
    print(f"Description length distribution plot saved to {description_length_dist_path}")
    
    # Save description length data to CSV
    description_lengths = data['Description_Length'].describe()
    description_lengths.to_csv(os.path.join(output_dir, 'description_length_summary.csv'), header=True)
    print(f"Description length summary data saved to CSV.")

# 3. Correlation Analysis (for numeric features, if available)
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_features) > 0:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[numeric_features].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    correlation_matrix_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(correlation_matrix_path)
    plt.show()
    print(f"Correlation matrix saved to {correlation_matrix_path}")
    
    # Save correlation matrix to CSV
    correlation_matrix = data[numeric_features].corr()
    correlation_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    print(f"Correlation matrix data saved to CSV.")
else:
    print("No numeric features found for correlation analysis.")

# 4. Brand Distribution Analysis
if 's:brand' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y='s:brand', data=data, order=data['s:brand'].value_counts().index[:10])
    plt.title('Top 10 Brands')
    brand_dist_path = os.path.join(output_dir, 'brand_distribution.png')
    plt.savefig(brand_dist_path)
    plt.show()
    print(f"Brand distribution plot saved to {brand_dist_path}")
    
    # Save brand distribution data to CSV
    brand_counts = data['s:brand'].value_counts().head(10)
    brand_counts.to_csv(os.path.join(output_dir, 'brand_distribution.csv'), header=True)
    print(f"Brand distribution data saved to CSV.")

# 5. Distribution of Brands within Top Categories
if 's:category' in data.columns and 's:brand' in data.columns:
    plt.figure(figsize=(12, 8))
    top_categories = data['s:category'].value_counts().index[:5]
    sns.countplot(y='s:brand', hue='s:category', data=data[data['s:category'].isin(top_categories)], order=data['s:brand'].value_counts().iloc[:10].index)
    plt.title('Top Brands by Top 5 Product Categories')
    brand_category_dist_path = os.path.join(output_dir, 'brand_category_distribution.png')
    plt.savefig(brand_category_dist_path)
    plt.show()
    print(f"Brand-Category distribution plot saved to {brand_category_dist_path}")
    
    # Save brand-category distribution data to CSV
    brand_category_counts = data[data['s:category'].isin(top_categories)]['s:brand'].value_counts().head(10)
    brand_category_counts.to_csv(os.path.join(output_dir, 'brand_category_distribution.csv'), header=True)
    print(f"Brand-Category distribution data saved to CSV.")

# 6. Description Length by Category
if 's:category' in data.columns and 'Description_Length' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='s:category', y='Description_Length', data=data, order=data['s:category'].value_counts().index[:10])
    plt.title('Description Length by Top 10 Categories')
    plt.xticks(rotation=90)
    desc_length_by_cat_path = os.path.join(output_dir, 'description_length_by_category.png')
    plt.savefig(desc_length_by_cat_path)
    plt.show()
    print(f"Description length by category plot saved to {desc_length_by_cat_path}")
    
    # Save description length by category data to CSV
    desc_length_by_category = data.groupby('s:category')['Description_Length'].describe().reset_index()
    desc_length_by_category.to_csv(os.path.join(output_dir, 'description_length_by_category.csv'), index=False)
    print(f"Description length by category data saved to CSV.")

print("Enhanced EDA completed and all outputs saved successfully.")

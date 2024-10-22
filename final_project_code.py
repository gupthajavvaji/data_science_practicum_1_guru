import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# URLs for the datasets
usda_organic_trends_url = "https://www.ers.usda.gov/webdocs/publications/44430/11009_eib58_1_.pdf?v="

# Downloading the USDA organic trends PDF
usda_response = requests.get(usda_organic_trends_url)

# Save the PDF to a file
with open('usda_organic_trends.pdf', 'wb') as file:
    file.write(usda_response.content)

print('USDA Organic Trends PDF downloaded successfully.')
import fitz

pdf_document = 'usda_organic_trends.pdf'
doc = fitz.open(pdf_document)

text = ""
for page_num in range(min(5, doc.page_count)):
    page = doc.load_page(page_num)
    text += page.get_text()

doc.close()
print(text[:1000])  # Printing the first 1000 characters for a quick overview

# Simulating customer data
np.random.seed(42)
n_customers = 1000

# Generating synthetic customer data
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.randint(18, 80, n_customers),
    'income': np.random.randint(20000, 200000, n_customers),
    'spending': np.random.randint(50, 1000, n_customers),
    'frequency': np.random.randint(1, 30, n_customers),
    'organic_preference': np.random.rand(n_customers)
}

df = pd.DataFrame(data)

# Normalize the data
scaler = StandardScaler()
features = ['age', 'income', 'spending', 'frequency', 'organic_preference']
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Performing K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='spending', y='frequency', hue='cluster', palette='viridis')
plt.title('Customer Segments based on Spending and Frequency')
plt.savefig('customer_segments.png')
plt.close()

# Train a Random Forest Classifier
X = df[features]
y = df['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Makeing predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
print("\
Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Customer Segmentation')
plt.savefig('feature_importance.png')
plt.close()

print("Customer segmentation and classification completed. Visualizations saved as PNG files.")

print("\
First few rows of the customer dataset:")
print(df.head())

# Basic statistics of the dataset
print("\
Basic statistics of the customer dataset:")
print(df.describe())

# Correlation matrix
correlation_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Customer Features')
plt.savefig('correlation_matrix.png')
plt.close()

print("Correlation matrix visualization saved as correlation_matrix.png")

# Simulating additional customer data
np.random.seed(42)
n_customers = 2000  # Increased number of customers for more data

# Generating synthetic customer data with additional features
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.randint(18, 80, n_customers),
    'income': np.random.randint(20000, 200000, n_customers),
    'spending': np.random.randint(50, 1000, n_customers),
    'frequency': np.random.randint(1, 30, n_customers),
    'organic_preference': np.random.rand(n_customers),
    'online_shopping': np.random.randint(0, 2, n_customers),  # New feature
    'loyalty_score': np.random.rand(n_customers) * 100  # New feature
}

df = pd.DataFrame(data)

# Normalize the data
scaler = StandardScaler()
features = ['age', 'income', 'spending', 'frequency', 'organic_preference', 'online_shopping', 'loyalty_score']
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Increased clusters for more granularity
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='spending', y='frequency', hue='cluster', palette='viridis')
plt.title('Customer Segments based on Spending and Frequency')
plt.savefig('customer_segments_expanded.png')
plt.close()

# Train a Random Forest Classifier
X = df[features]
y = df['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
print("\
Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Customer Segmentation')
plt.savefig('feature_importance_expanded.png')
plt.close()

print("Expanded customer segmentation and classification completed. Visualizations saved as PNG files.")

# Display the first few rows of the dataset
print("\
First few rows of the expanded customer dataset:")
print(df.head())

# Basic statistics of the dataset
print("\
Basic statistics of the expanded customer dataset:")
print(df.describe())

# Correlation matrix
correlation_matrix = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Expanded Customer Features')
plt.savefig('correlation_matrix_expanded.png')
plt.close()

print("Correlation matrix visualization saved as correlation_matrix_expanded.png")

# Loading the Instacart data files into dataframes
order_products_train = pd.read_csv('order_products__train.csv')
aisles = pd.read_csv('aisles.csv')
orders = pd.read_csv('orders.csv')
sample_submission = pd.read_csv('sample_submission.csv')
departments = pd.read_csv('departments.csv')
products = pd.read_csv('products.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')

# Displaying the first few rows of each dataframe to verify loading
print("order_products_train:")
print(order_products_train.head())

print("\
aisles:")
print(aisles.head())

print("\
orders:")
print(orders.head())

print("\
sample_submission:")
print(sample_submission.head())

print("\
departments:")
print(departments.head())

print("\
products:")
print(products.head())

print("\
order_products_prior:")
print(order_products_prior.head())

# Loading the data required
order_products_train = pd.read_csv('order_products__train.csv')
aisles = pd.read_csv('aisles.csv')
orders = pd.read_csv('orders.csv')
departments = pd.read_csv('departments.csv')
products = pd.read_csv('products.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')

# Merge relevant dataframes
orders_products = pd.concat([order_products_train, order_products_prior])
orders_products = orders_products.merge(products, on='product_id')
orders_products = orders_products.merge(aisles, on='aisle_id')
orders_products = orders_products.merge(departments, on='department_id')
orders_products = orders_products.merge(orders, on='order_id')

# Display info about the merged dataframe
print("Merged dataframe info:")
orders_products.info()

# Check for missing values
print("\
Missing values:")
print(orders_products.isnull().sum())

# Handling missing values
orders_products['days_since_prior_order'].fillna(0, inplace=True)

# Display summary statistics
print("\
Summary statistics:")
print(orders_products.describe())

# Saving the cleaned and merged dataframe
orders_products.to_csv('cleaned_merged_instacart_data.csv', index=False)
print("\
Cleaned and merged data saved to 'cleaned_merged_instacart_data.csv'")

# Display the first few rows of the merged dataframe
print("\
First few rows of the merged dataframe:")
print(orders_products.head())

# Generating visualizations to explore the data

# Set the style of seaborn
sns.set_theme(style="whitegrid")

# Plot the distribution of order days of the week
plt.figure(figsize=(10, 6))
sns.countplot(x='order_dow', data=orders_products, palette='viridis')
plt.title('Distribution of Orders by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Orders')
plt.show()

# Plot the distribution of order hours of the day
plt.figure(figsize=(10, 6))
sns.countplot(x='order_hour_of_day', data=orders_products, palette='viridis')
plt.title('Distribution of Orders by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Orders')
plt.show()

# Plot the top 10 most ordered products
top_products = orders_products['product_name'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Most Ordered Products')
plt.xlabel('Number of Orders')
plt.ylabel('Product Name')
plt.show()

# Plot the top 10 aisles
top_aisles = orders_products['aisle'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_aisles.values, y=top_aisles.index, palette='viridis')
plt.title('Top 10 Aisles')
plt.xlabel('Number of Orders')
plt.ylabel('Aisle')
plt.show()

# Plot the top 10 departments
top_departments = orders_products['department'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_departments.values, y=top_departments.index, palette='viridis')
plt.title('Top 10 Departments')
plt.xlabel('Number of Orders')
plt.ylabel('Department')
plt.show()

#recommendations
print(f"Orders shape: {orders.shape}")
print(f"Products shape: {products.shape}")
print(f"Order_products shape: {order_products_prior.shape}")

# Merge the datasets
df = order_products_prior.merge(orders, on='order_id')

# Create user and product label encoders
user_le = LabelEncoder()
product_le = LabelEncoder()

df['user_encoded'] = user_le.fit_transform(df['user_id'])
df['product_encoded'] = product_le.fit_transform(df['product_id'])

# Create a sparse user-item matrix
user_item_matrix = csr_matrix((np.ones(len(df)), 
                               (df['user_encoded'], df['product_encoded'])))

print("Sparse user-item interaction matrix created.")
print(f"Matrix shape: {user_item_matrix.shape}")
print(f"Matrix memory usage: {user_item_matrix.data.nbytes / 1e6:.2f} MB")

# Implement item-based collaborative filtering
def get_item_similarities(item_matrix, n_items=1000):
    # Calculate similarities for a subset of items to save memory
    similarities = cosine_similarity(item_matrix.T[:n_items])
    return similarities

item_similarities = get_item_similarities(user_item_matrix)

print("Item similarities calculated.")
print(f"Similarity matrix shape: {item_similarities.shape}")

def get_recommendations(user_id, N=5, n_items=1000):
    user_index = user_le.transform([user_id])[0]
    user_items = user_item_matrix[user_index, :n_items].toarray().flatten()
    scores = item_similarities.dot(user_items)
    top_items = np.argsort(scores)[::-1][:N]
    return product_le.inverse_transform(top_items)

# Get recommendations for a sample user
sample_user = df['user_id'].iloc[0]
recommended_product_ids = get_recommendations(sample_user)

print(f"\
Top 5 recommendations for user {sample_user}:")
for i, product_id in enumerate(recommended_product_ids, 1):
    product_name = products[products['product_id'] == product_id]['product_name'].values[0]
    print(f"{i}. {product_name}")

print("Done")

# feature
# Merge datasets
df = pd.merge(orders, order_products_prior, on='order_id')
print(f"Merged dataset shape: {df.shape}")

# Check for missing values
missing_values = df.isnull().sum()
print("\
Missing values in each column:")
print(missing_values)

# Calculate percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
print("\
Percentage of missing values in each column:")
print(missing_percentage)

# Handle missing values
df['days_since_prior_order'].fillna(df['days_since_prior_order'].median(), inplace=True)

print("Missing values handled.")
# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Prepare features and target variable
features = ['order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order']
target = 'reordered'

# Train a simpler Random Forest model
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(train_data[features], train_data[target])

# Make predictions
y_pred = rf_model.predict(test_data[features])
y_pred_proba = rf_model.predict_proba(test_data[features])[:, 1]

# Calculate metrics
precision = precision_score(test_data[target], y_pred)
recall = recall_score(test_data[target], y_pred)
f1 = f1_score(test_data[target], y_pred)
roc_auc = roc_auc_score(test_data[target], y_pred_proba)

print("Advanced Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\
Feature Importance:")
print(feature_importance)

#personalized market
# Creating a function for personalized product recommendations
def get_organic_recommendations(product_id, cosine_sim, products_df):
    idx = products_df.index[products_df['product_id'] == product_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar products
    product_indices = [i[0] for i in sim_scores]
    return products_df.iloc[product_indices]

# Preprocess product names
products['product_name'] = products['product_name'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['product_name'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Example of getting recommendations for an organic product
organic_product_id = products[products['product_name'].str.contains('organic', case=False)]['product_id'].iloc[0]
recommendations = get_organic_recommendations(organic_product_id, cosine_sim, products)

print("Personalized recommendations for organic product (ID: {}):\
".format(organic_product_id))
print(recommendations[['product_id', 'product_name']].head())

# Add a flag for organic products
products['is_organic'] = products['product_name'].str.contains('organic', case=False)

# Calculate the percentage of organic products
organic_percentage = (products['is_organic'].sum() / len(products)) * 100

print("\
Percentage of organic products: {:.2f}%".format(organic_percentage))

# Displaying some statistics about organic products
print("\
Organic product statistics:")
print(products[products['is_organic']]['aisle_id'].describe())
# Merge datasets
df = order_products_prior.merge(products, on='product_id')
df = df.merge(orders[['order_id', 'user_id']], on='order_id')

# Calculating organic preference for each user
df['is_organic'] = df['product_name'].str.contains('organic', case=False).astype(int)
user_organic_pref = df.groupby('user_id').agg({
    'is_organic': 'mean',
    'order_id': 'count'
}).reset_index()
user_organic_pref.columns = ['user_id', 'organic_preference', 'total_orders']

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(user_organic_pref[['organic_preference', 'total_orders']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
user_organic_pref['cluster'] = kmeans.fit_predict(features)

# Analyze clusters
cluster_analysis = user_organic_pref.groupby('cluster').agg({
    'organic_preference': 'mean',
    'total_orders': 'mean',
    'user_id': 'count'
}).reset_index()
cluster_analysis.columns = ['cluster', 'avg_organic_preference', 'avg_total_orders', 'user_count']

print("Customer Segmentation Results:")
print(cluster_analysis)

# Visualize clusters
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
scatter = plt.scatter(user_organic_pref['organic_preference'],
                      user_organic_pref['total_orders'],
                      c=user_organic_pref['cluster'],
                      cmap='viridis',
                      alpha=0.5)
plt.colorbar(scatter)
plt.xlabel('Organic Preference')
plt.ylabel('Total Orders')
plt.title('Customer Segments based on Organic Preference and Order Frequency')
plt.savefig('customer_segments.png')
plt.close()

print("\
Customer segmentation visualization saved as 'customer_segments.png'")
# Analyze purchasing patterns
organic_vs_nonorganic = df.groupby('is_organic').agg({
    'order_id': 'count',
    'reordered': 'mean'
}).reset_index()
organic_vs_nonorganic.columns = ['is_organic', 'total_orders', 'reorder_rate']
organic_vs_nonorganic['order_percentage'] = organic_vs_nonorganic['total_orders'] / organic_vs_nonorganic['total_orders'].sum() * 100

print("Organic vs Non-Organic Purchasing Patterns:")
print(organic_vs_nonorganic)

# Analyze top organic and non-organic products
top_products = df.groupby(['product_name', 'is_organic']).agg({
    'order_id': 'count'
}).reset_index().sort_values('order_id', ascending=False)

print("\
Top 5 Organic Products:")
print(top_products[top_products['is_organic'] == 1].head())

print("\
Top 5 Non-Organic Products:")
print(top_products[top_products['is_organic'] == 0].head())

# Analyzing organic purchases by day of week
df['order_dow'] = orders.set_index('order_id').loc[df['order_id'], 'order_dow'].values
organic_by_dow = df[df['is_organic'] == 1].groupby('order_dow').agg({
    'order_id': 'count'
}).reset_index()
organic_by_dow['percentage'] = organic_by_dow['order_id'] / organic_by_dow['order_id'].sum() * 100

print("\
Organic Purchases by Day of Week:")
print(organic_by_dow)

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Final Project Report: Customer Segmentation and Personalized Marketing for Organic Groceries', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

# Introduction
pdf.multi_cell(0, 10, """
This report presents the findings from the project on customer segmentation and personalized marketing for organic groceries. The analysis focuses on understanding customer preferences for organic products and developing strategies to enhance marketing efforts.
""")

# Customer Segmentation
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Customer Segmentation', 0, 1)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, """
We performed customer segmentation based on their preference for organic products and total order frequency. The analysis identified four distinct customer segments:
- Cluster 0: Moderate organic buyers with average order frequency
- Cluster 1: High-value customers with strong organic preference
- Cluster 2: Low organic preference with low order frequency
- Cluster 3: Strong organic preference with average order frequency
""")
pdf.image('customer_segments.png', x=10, y=None, w=180)

# Purchasing Patterns
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Purchasing Patterns', 0, 1)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, """
The analysis of purchasing patterns revealed that organic products account for 31.6% of all orders, with a higher reorder rate compared to non-organic products. The top organic and non-organic products were identified, providing insights into customer preferences.
""")

# Marketing Strategies
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Marketing Strategies', 0, 1)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, """
Based on the insights gained, several marketing strategies were developed to enhance the promotion of organic products:
1. Targeted Promotions
2. Product Recommendations
3. Cross-selling Strategies
4. Day-of-Week Promotions
5. Loyalty Program
6. Educational Marketing
7. Reorder Reminders
8. Seasonal Campaigns
""")

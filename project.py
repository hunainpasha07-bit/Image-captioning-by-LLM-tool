import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("ecommerce_sales.csv")

# Step 2: Basic Cleaning
df.dropna(inplace=True)   # remove missing values
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Step 3: Monthly Sales Trend
monthly_sales = df.groupby(df['OrderDate'].dt.to_period("M"))['Sales'].sum()
monthly_sales.plot(kind='bar', figsize=(10,5), color='skyblue')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# Step 4: Top Products
top_products = df.groupby('ProductName')['Sales'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Products by Sales")
plt.show()

# Step 5: Customer Segmentation
customer_sales = df.groupby('CustomerID')['Sales'].sum()
print("Average spend per customer:", customer_sales.mean())

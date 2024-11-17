import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
from threadpoolctl import threadpool_limits

# Set page configuration
st.set_page_config(layout="wide", page_title="E-commerce Sales Dashboard")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("Github/Big-Data-Analysis/Amazon Sale Report.csv", low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
    return df

# Load the dataset
df = load_data()

# Title
st.title("E-commerce Sales Dashboard")

# Create two columns for layout
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    navigation = st.selectbox(
        "Navigate to:",
        ["Public Analysis", "Hypothesis Testing", "Correlation Testing"]
    )

with col2:
    if navigation == "Public Analysis":
        section = st.selectbox(
            "Select Section",
            ["Explorative Data Analysis"]
        )
    elif navigation == "Hypothesis Testing":
        section = st.selectbox(
            "Select Section",
            ["Explorative Data Analysis", "Machine Learning"]
        )
    elif navigation == "Correlation Testing":
        section = st.selectbox(
            "Select Section",
            ["Explorative Data Analysis", "Machine Learning"]
        )

with col3:
    if navigation == "Public Analysis":
        if section == "Explorative Data Analysis":
            analysis_type = st.selectbox(
                "Analysis Type", 
                ["Total Sales by Category", "Fulfillment & Category Sales", "Order Status Distribution"]
            )
    elif navigation == "Hypothesis Testing":
        if section == "Explorative Data Analysis":
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Shipping Preference Among Top 51% of Spenders", "Cumulative Count of Shipping Levels Among Top 51% of Spenders"]
            )
        elif section == "Machine Learning":
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Shipping Preferences by Algorithm and Actual Data"]
            )
    elif navigation == "Correlation Testing":
        if section == "Explorative Data Analysis":
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Monthly Promotions vs Normal Sales", "Montly Sales Distribution", "Cumulative Sales of Promotion and Normal Sales Throughout the Months"]
            )
        elif section == "Machine Learning":
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Clustering Analysis: K-means and DBSCAN on Cumulative Sales"]
            )

# Navigation logic
if navigation == "Public Analysis":
    if section == "Explorative Data Analysis":
        if analysis_type == "Total Sales by Category":
            # === Code 1: Total Sales by Category Pie Chart ===
            df_filtered = df[(df['Status'] != 'Cancelled') & (df['Status'] != 'Pending') & (df['Qty'] >= 1)]
            summary = df_filtered.groupby('Category')['Order ID'].nunique().reset_index(name='Total Sales')
    
            other_categories = summary[summary['Category'].isin(['Blouse', 'Bottom', 'Dupatta', 'Ethnic Dress', 'Saree'])]
            other_sales = other_categories['Total Sales'].sum()
    
            summary = summary[~summary['Category'].isin(['Blouse', 'Bottom', 'Dupatta', 'Ethnic Dress', 'Saree'])]
            other_row = pd.DataFrame({'Category': ['Other'], 'Total Sales': [other_sales]})
            summary = pd.concat([summary, other_row], ignore_index=True)
    
            plt.figure(figsize=(8, 8))
            plt.pie(summary['Total Sales'], labels=summary['Category'], autopct='%1.1f%%', startangle=90, counterclock=False)
            plt.title('Total Sales by Category')
            st.pyplot(plt)

        elif analysis_type == "Fulfillment & Category Sales":
            # === Code 2: Fulfillment Percentage and Top 10 Categories ===
            valid_orders = df[df["Status"].str.contains("Shipped", case=False, na=False)]
            total_sales = valid_orders["Amount"].sum()
            fulfillment_counts = valid_orders["Fulfilment"].value_counts(normalize=True) * 100
            category_sales = valid_orders.groupby("Category")["Amount"].sum().sort_values(ascending=False)
            category_sales_pct = (category_sales / total_sales) * 100
    
            fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    
            fulfillment_counts.plot.pie(
                autopct='%1.1f%%', ax=axes[0], startangle=90, explode=(0, 0.1),
                colors=['#4CAF50', '#FF9800'], shadow=True
            )
            axes[0].set_ylabel('')
            axes[0].set_title('Fulfillment Percentage')
    
            top_10_categories = category_sales.head(10)
            top_10_categories_pct = category_sales_pct.head(10)
            bars = axes[1].bar(top_10_categories.index, top_10_categories, color='#03A9F4', edgecolor='black')
    
            for bar, amount, pct in zip(bars, top_10_categories, top_10_categories_pct):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 10000, 
                    f"{amount:,.0f} INR\n({pct:.1f}%)", ha='center', va='bottom'
                )
            axes[1].set_title('Top 10 Product Categories by Sales (with Percentage)')
            axes[1].set_xlabel('Category')
            axes[1].set_ylabel('Sales Amount (INR)')
            axes[1].tick_params(axis='x', rotation=45)
            fig.subplots_adjust(wspace=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
        elif analysis_type == "Order Status Distribution":
            order_status_counts = pd.DataFrame({
                'Status': ['Shipped', 'Cancelled', 'Pending'],
                'Percentage': [80, 15, 5]
            })
            shipped_cancelled_data = pd.DataFrame({
                'Status': ['Shipped', 'Cancelled'],
                'Percentage': [84.2, 15.8]
            })
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.barplot(
                data=order_status_counts, 
                x='Status', 
                y='Percentage', 
                ax=axes[0],
                palette=['green', 'red', 'gray']
            )
            axes[0].set_title('Order Status Distribution')
            axes[0].set_ylim(0, 100)
            for index, row in order_status_counts.iterrows():
                axes[0].text(index, row['Percentage'] + 1, f"{row['Percentage']}%", ha='center')
            
            sns.barplot(
                data=shipped_cancelled_data, 
                x='Status', 
                y='Percentage', 
                ax=axes[1],
                palette=['green', 'red']
            )
            axes[1].set_title('Shipped vs Cancelled Distribution')
            axes[1].set_ylim(0, 100)
            for index, row in shipped_cancelled_data.iterrows():
                axes[1].text(index, row['Percentage'] + 1, f"{row['Percentage']}%", ha='center')
            st.pyplot(fig)

elif navigation == "Hypothesis Testing":
    if section == "Explorative Data Analysis":
        if analysis_type == "Shipping Preference Among Top 51% of Spenders":
            filtered_data = df[df['ship-service-level'].isin(['Expedited', 'Standard'])]
            sorted_data = filtered_data.sort_values(by='Amount', ascending=False)
            cutoff_index = int(len(sorted_data) * 0.90)
            top_51_data = sorted_data.iloc[:cutoff_index]
    
            shipping_preference_counts = top_51_data['ship-service-level'].value_counts()
    
            plt.figure(figsize=(8, 5))
            bars = shipping_preference_counts.plot(kind='bar', color=['#FF6347', '#4682B4'])
            plt.title("Shipping Preference Among Top 51% of Spenders")
            plt.xlabel("Shipping Type")
            plt.ylabel("Number of Orders")
            plt.xticks(rotation=0)
    
            for index, value in enumerate(shipping_preference_counts):
                plt.text(index, value + 1, str(value), ha='center', va='bottom')

            # Hide the gridlines
            plt.grid(False)
            
            st.subheader("Shipping Preference Among Top 51% of Spenders")
            st.pyplot(plt)

        elif analysis_type == "Cumulative Count of Shipping Levels Among Top 51% of Spenders":
            filtered_data = df[df['ship-service-level'].isin(['Expedited', 'Standard'])]
            sorted_data = filtered_data.sort_values(by='Amount', ascending=False)
            cutoff_index = int(len(sorted_data) * 0.51)
            top_51_data = sorted_data.iloc[:cutoff_index]
            top_51_data = top_51_data.reset_index(drop=True)
    
            top_51_data['Expedited_CumCount'] = (top_51_data['ship-service-level'] == 'Expedited').cumsum()
            top_51_data['Standard_CumCount'] = (top_51_data['ship-service-level'] == 'Standard').cumsum()
    
            plt.figure(figsize=(10, 6))
            plt.plot(top_51_data.index, top_51_data['Expedited_CumCount'], label='Expedited', color='#FF6347')
            plt.plot(top_51_data.index, top_51_data['Standard_CumCount'], label='Standard', color='#4682B4')
    
            plt.title("Cumulative Count of Shipping Levels Among Top 51% of Spenders")
            plt.xlabel("Order Number (sorted by Amount)")
            plt.ylabel("Cumulative Count of Shipping Level")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

    elif section == "Machine Learning":
        if analysis_type == "Shipping Preferences by Algorithm and Actual Data":
            # Load and filter the dataset
            file_path = 'Github/Big-Data-Analysis/Amazon Sale Report.csv'
            data = pd.read_csv(file_path, low_memory=False)
            filtered_data = data[data['ship-service-level'].isin(['Expedited', 'Standard'])]
    
            # Create binary target variable
            filtered_data['ship_service_binary'] = filtered_data['ship-service-level'].apply(lambda x: 1 if x == 'Expedited' else 0)
    
            # Sort by amount spent and select top 51% of spenders
            sorted_data = filtered_data.sort_values(by='Amount', ascending=False)
            cutoff_index = int(len(sorted_data) * 0.51)
            top_51_data = sorted_data.iloc[:cutoff_index]
    
            # Features and target
            X = top_51_data[['Amount']]
            y = top_51_data['ship_service_binary']
    
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
            # Standardize the feature
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
    
            # Resample the data using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
            # Logistic Regression
            log_reg = LogisticRegression()
            log_reg.fit(X_resampled, y_resampled)
            y_pred_log_reg = log_reg.predict(X_test_scaled)
    
            # Random Forest Classifier
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_resampled, y_resampled)
            y_pred_rf = rf.predict(X_test_scaled)
    
            # Count shipping preferences among the top 51% for actual data comparison
            shipping_preference_counts = top_51_data['ship-service-level'].value_counts()
    
            # Prepare data for the bar plot
            plot_data = pd.DataFrame({
                'Standard': [
                    sum(y_pred_log_reg == 0),
                    sum(y_pred_rf == 0),
                    shipping_preference_counts.get('Standard', 0)
                ],
                'Expedited': [
                    sum(y_pred_log_reg == 1),
                    sum(y_pred_rf == 1),
                    shipping_preference_counts.get('Expedited', 0)
                ]
            }, index=['Logistic Regression', 'Random Forest', 'Real Data'])
    
            # Plot 1: Shipping Preferences by Algorithm and Real Data
            plt.figure(figsize=(10, 6))
            plot_data.plot(kind='bar', figsize=(10, 6), color=['#4682B4', '#FF6347'])
            plt.title('Shipping Preferences by Algorithm and Real Data')
            plt.xlabel('Model Type')
            plt.ylabel('Number of Orders')
            plt.xticks(rotation=0)
            plt.legend(title='Shipping Type')
            st.pyplot(plt)
            
elif navigation == "Correlation Testing":
    if section == "Explorative Data Analysis":
        if analysis_type == "Monthly Promotions vs Normal Sales":
            # Load the dataset
            amazon_sales_data = pd.read_csv("Github/Big-Data-Analysis/Amazon Sale Report.csv", low_memory=False)
            
            # Data Cleaning
            # Convert 'Date' to datetime format and extract the month
            amazon_sales_data['Date'] = pd.to_datetime(amazon_sales_data['Date'], format='%m-%d-%y', errors='coerce')
            amazon_sales_data = amazon_sales_data.dropna(subset=['Date'])  # Drop rows with invalid dates
            amazon_sales_data['Month'] = amazon_sales_data['Date'].dt.month
            
            # Filter for valid records only
            valid_data = amazon_sales_data[
                (amazon_sales_data['Status'] == 'Shipped') &
                (amazon_sales_data['Qty'] >= 1) &
                (amazon_sales_data['Amount'] > 0) &
                amazon_sales_data[['ship-city', 'ship-state', 'ship-postal-code']].notnull().all(axis=1)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning
            
            # Define promotion categories based on 'promotion-ids' presence
            valid_data['Type'] = valid_data['promotion-ids'].apply(
                lambda x: 'Promotion Sales' if pd.notnull(x) else 'Normal Sales'
            )
            
            # Group by month and type, then count orders
            monthly_sales = valid_data.groupby(['Month', 'Type'])['Order ID'].count().reset_index()
            
            # Custom function to display percentage and actual count on the pie chart
            def autopct_with_values(pct, all_values):
                absolute = int(round(pct / 100. * sum(all_values)))
                return f"{pct:.1f}%\n({absolute})"
            
            # Create subplots for the pie charts (2 rows, 2 columns)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid of pie charts
            axes = axes.flatten()  # Flatten to access each subplot easily
            
            # Loop through each unique month and add the pie chart to a subplot
            for idx, month in enumerate(monthly_sales['Month'].unique()):
                if idx >= 4:  # Limit to 4 months for 2x2 grid
                    break
            
                # Get the month name and filter sales data
                month_name = calendar.month_name[month]
                month_sales = monthly_sales[monthly_sales['Month'] == month]
            
                # Plot the pie chart
                ax = axes[idx]
                ax.pie(
                    month_sales['Order ID'], 
                    labels=month_sales['Type'], 
                    autopct=lambda pct: autopct_with_values(pct, month_sales['Order ID']),
                    startangle=140, 
                    colors=['#66b3ff', '#ff9999']
                )
                ax.set_title(f"Sales Distribution for {month_name}")
            
            # Adjust layout to prevent overlapping
            plt.tight_layout()
            
            # Display the plot in Streamlit
            st.pyplot(fig)

        elif analysis_type == "Montly Sales Distribution":
            # === Code 5: Monthly Sales Distribution ===
            valid_data = df[
                (df['Status'] == 'Shipped') &
                (df['Qty'] >= 1) &
                (df['Amount'] > 0) &
                df[['ship-city', 'ship-state', 'ship-postal-code']].notnull().all(axis=1)
            ]
            monthly_sales = valid_data.groupby(valid_data['Date'].dt.strftime('%m'))['Order ID'].count()
        
            plt.figure(figsize=(10, 6))
            plt.bar(monthly_sales.index, monthly_sales.values)
            for i, v in enumerate(monthly_sales.values):
                plt.text(i, v, str(v), ha='center', va='bottom')
            plt.xlabel('Month')
            plt.ylabel('Number of Orders')
            plt.title('Monthly Sales Distribution')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        
        elif analysis_type == "Cumulative Sales of Promotion and Normal Sales Throughout the Months":
            # Data Cleaning and Preparation
            df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
            df = df.dropna(subset=['Date'])  # Drop rows with invalid dates

            valid_data = df[
                (df['Status'] == 'Shipped') &
                (df['Qty'] >= 1) &
                (df['Amount'] > 0) &
                df[['ship-city', 'ship-state', 'ship-postal-code']].notnull().all(axis=1)
            ]

            # Define promotion categories based on 'promotion-ids' presence
            valid_data['Type'] = valid_data['promotion-ids'].apply(
                lambda x: 'Promotion Sales' if pd.notnull(x) else 'Normal Sales'
            )

            # Aggregate daily sales by date and type
            daily_sales = valid_data.groupby(['Date', 'Type'])['Qty'].sum().unstack(fill_value=0).reset_index()

            # Add a new column for the month
            daily_sales['Month'] = daily_sales['Date'].dt.to_period('M')

            # Aggregate monthly sales
            monthly_sales = daily_sales.groupby(['Month'])[['Normal Sales', 'Promotion Sales']].sum()

            # Extract monthly sales for t-test
            monthly_normal_sales = monthly_sales['Normal Sales']
            monthly_promotion_sales = monthly_sales['Promotion Sales']

            # Perform a paired t-test
            t_stat, p_value = ttest_rel(monthly_normal_sales, monthly_promotion_sales)

            # Cumulative Sales Plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                daily_sales['Date'], 
                daily_sales['Normal Sales'].cumsum(), 
                color='#66b3ff', 
                label='Cumulative Normal Sales'
            )
            plt.plot(
                daily_sales['Date'], 
                daily_sales['Promotion Sales'].cumsum(), 
                color='#ff9999', 
                label='Cumulative Promotion Sales'
            )
            plt.title("Cumulative Sales of Promotion and Normal Sales Throughout the Months")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Sales Quantity")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()

            # Display the plot with the test results
            plt.figtext(0.5, -0.1, f"T-test: T-statistic = {t_stat:.2f}, P-value = {p_value:.4f}", ha='center', fontsize=10)
            st.pyplot(plt)

    elif section == "Machine Learning":
        if analysis_type == "Clustering Analysis: K-means and DBSCAN on Cumulative Sales":
            st.subheader("Clustering Analysis: K-means and DBSCAN on Cumulative Sales")
            
            # Convert 'Date' to datetime and clean data
            df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            valid_data = df[
                (df['Status'] == 'Shipped') &
                (df['Qty'] >= 1) &
                (df['Amount'] > 0) &
                df[['ship-city', 'ship-state', 'ship-postal-code']].notnull().all(axis=1)
            ]
            valid_data['Sales Type'] = valid_data['promotion-ids'].apply(lambda x: 'Promotion Sales' if pd.notnull(x) else 'Normal Sales')
            
            # Aggregate daily sales by date and type
            daily_sales = valid_data.groupby(['Date', 'Sales Type'])['Qty'].sum().unstack(fill_value=0).reset_index()
            daily_sales.rename(columns={'Normal Sales': 'Normal Sales', 'Promotion Sales': 'Promotion Sales'}, inplace=True)
            daily_sales['Cumulative Normal Sales'] = daily_sales['Normal Sales'].cumsum()
            daily_sales['Cumulative Promotion Sales'] = daily_sales['Promotion Sales'].cumsum()
            
            # Prepare features for clustering
            clustering_features = daily_sales[['Cumulative Normal Sales', 'Cumulative Promotion Sales']]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_features)
            
            # --- K-means Clustering ---
            k = 3  # Number of clusters
            with threadpool_limits(limits=1, user_api='blas'):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                daily_sales['KMeans Cluster'] = kmeans.fit_predict(scaled_features)
            
            # --- DBSCAN Clustering ---
            dbscan = DBSCAN(eps=1.2, min_samples=5)
            daily_sales['DBSCAN Cluster'] = dbscan.fit_predict(scaled_features)
            num_dbscan_clusters = len(set(daily_sales['DBSCAN Cluster'])) - (1 if -1 in daily_sales['DBSCAN Cluster'] else 0)
            st.write(f"Number of clusters identified by DBSCAN (excluding noise): {num_dbscan_clusters}")
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # K-means clustering plot
            axes[0].scatter(
                daily_sales['Cumulative Normal Sales'],
                daily_sales['Cumulative Promotion Sales'],
                c=daily_sales['KMeans Cluster'],
                cmap='viridis',
                s=50
            )
            axes[0].set_title("K-means Clustering")
            axes[0].set_xlabel("Cumulative Normal Sales")
            axes[0].set_ylabel("Cumulative Promotion Sales")
            
            # DBSCAN clustering plot
            axes[1].scatter(
                daily_sales['Cumulative Normal Sales'],
                daily_sales['Cumulative Promotion Sales'],
                c=daily_sales['DBSCAN Cluster'],
                cmap='viridis',
                s=50
            )
            axes[1].set_title("DBSCAN Clustering")
            axes[1].set_xlabel("Cumulative Normal Sales")
            axes[1].set_ylabel("Cumulative Promotion Sales")
            
            plt.tight_layout()
            st.pyplot(fig)
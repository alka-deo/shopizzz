"""
Shopizz.com Analytics Dashboard
A comprehensive ML analytics platform for customer behavior analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from scipy import stats

# Attempt to import XGBoost, handle gracefully if not installed
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

# Page Configuration
st.set_page_config(
    page_title="Shopizz Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        border: 2px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
        border-color: #4CAF50;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION & LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic customer data for Shopizz.com"""
    np.random.seed(42)
    
    # Demographics
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    genders = ['Male', 'Female', 'Non-binary']
    locations = ['Metro', 'Tier-1', 'Tier-2', 'Tier-3']
    income_levels = ['<25K', '25-50K', '50-75K', '75-100K', '100-150K', '>150K']
    
    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
        'age_group': np.random.choice(age_groups, n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'gender': np.random.choice(genders, n_samples, p=[0.48, 0.48, 0.04]),
        'location': np.random.choice(locations, n_samples, p=[0.3, 0.25, 0.25, 0.2]),
        'income': np.random.choice(income_levels, n_samples, p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]),
        'monthly_spending': np.random.choice(['<1K', '1-3K', '3-5K', '5-10K', '>10K'], n_samples, 
                                            p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'sustainability': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_samples,
                                          p=[0.15, 0.35, 0.35, 0.15]),
        'artisan_support': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_samples,
                                           p=[0.2, 0.3, 0.35, 0.15]),
        'shopping_style': np.random.choice(['Budget', 'Balanced', 'Premium', 'Luxury'], n_samples,
                                          p=[0.25, 0.4, 0.25, 0.1]),
        'categories_interested': np.random.randint(1, 6, n_samples),
        'features_wanted': np.random.randint(1, 8, n_samples),
    }
    
    # Generate willingness to pay (5 classes for classification)
    # Scale based on monthly spending index
    spending_map = ['<1K', '1-3K', '3-5K', '5-10K', '>10K']
    base_wtp = (
        pd.Series(data['monthly_spending']).apply(lambda x: spending_map.index(x)) * 0.3 +
        np.random.rand(n_samples) * 2
    )
    
    wtp_classes = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    data['willingness_to_pay_class'] = pd.cut(base_wtp, bins=5, labels=wtp_classes)
    data['willingness_to_pay_numeric'] = base_wtp
    
    # Binary classification target
    data['will_subscribe'] = (base_wtp > 2.5).astype(int)
    
    # Association rule items (shopping cart items)
    fashion_items = ['Dress', 'Shirt', 'Jeans', 'Shoes', 'Accessories']
    home_items = ['Cushions', 'Curtains', 'Decor', 'Furniture']
    wellness_items = ['Skincare', 'Haircare', 'Supplements', 'Fitness']
    
    all_items = fashion_items + home_items + wellness_items
    
    cart_items = []
    for _ in range(n_samples):
        n_items = np.random.randint(2, 6)
        items = np.random.choice(all_items, n_items, replace=False)
        cart_items.append('|'.join(items))
    
    data['cart_items'] = cart_items
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_data_from_github(url):
    """Load data from GitHub raw URL"""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return None

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Convert dataframe to Excel for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_data(df):
    """Preprocess data for ML models"""
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['age_group', 'gender', 'location', 'income', 'monthly_spending',
                       'sustainability', 'artisan_support', 'shopping_style']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            # Ensure the column is treated as string for consistent encoding
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    return df_processed, label_encoders

def get_feature_columns(df):
    """Get encoded feature columns for modeling"""
    return [col for col in df.columns if col.endswith('_encoded')] + \
           [col for col in df.columns if col in ['categories_interested', 'features_wanted']]

# =============================================================================
# TAB 1: ASSOCIATION RULE MINING
# =============================================================================

def association_rules_tab(df):
    st.header("🔗 Association Rule Mining")
    st.markdown("Discover patterns in customer shopping behavior")
    
    # Error Check for 'cart_items' column
    if 'cart_items' not in df.columns:
        st.error("⚠️ The DataFrame is missing the required column **'cart_items'** for Association Rule Mining. Please use the Sample Data or upload a file with transaction items in a column named 'cart_items'.")
        return
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01,
                               help="Minimum frequency of itemset")
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.7, 0.05,
                                  help="Minimum probability of rule")
    with col3:
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1,
                            help="Minimum improvement over random")
    
    if st.button("🔍 Mine Association Rules", type="primary"):
        with st.spinner("Mining association rules..."):
            # Prepare transaction data
            transactions = []
            for items in df['cart_items']:
                if pd.notna(items):
                    # Robustly handle different delimiters if necessary, but pipe is assumed here
                    transactions.append(items.split('|'))
            
            # Encode transactions
            te = TransactionEncoder()
            try:
                te_ary = te.fit(transactions).transform(transactions)
            except ValueError:
                st.warning("Transaction data is empty or malformed.")
                return
                
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Apply Apriori
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                st.warning("No frequent itemsets found. Try lowering the support threshold.")
                return
            
            # Generate rules
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                     min_threshold=min_confidence)
            rules = rules[rules['lift'] >= min_lift]
            
            if len(rules) == 0:
                st.warning("No rules found. Try adjusting the thresholds.")
                return
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequent Itemsets", len(frequent_itemsets))
            with col2:
                st.metric("Association Rules", len(rules))
            with col3:
                st.metric("Avg Lift", f"{rules['lift'].mean():.2f}")
            
            # Top rules
            st.subheader("📊 Top 10 Rules by Lift")
            top_rules = rules.nlargest(10, 'lift').reset_index(drop=True)
            
            display_rules = top_rules.copy()
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.dataframe(
                display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                .style.background_gradient(subset=['lift'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Support vs Confidence")
                fig = px.scatter(rules, x='support', y='confidence', size='lift',
                               color='lift', hover_data=['antecedents', 'consequents'],
                               title='Association Rules',
                               labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Top Rules by Metric")
                metric_choice = st.selectbox("Select Metric", ['lift', 'confidence', 'support'])
                top_n = rules.nlargest(10, metric_choice)
                
                fig = go.Figure(go.Bar(
                    x=top_n[metric_choice],
                    y=[f"Rule {i+1}" for i in range(len(top_n))],
                    orientation='h',
                    marker=dict(color=top_n[metric_choice], colorscale='Viridis')
                ))
                fig.update_layout(
                    title=f'Top 10 Rules by {metric_choice.capitalize()}',
                    xaxis_title=metric_choice.capitalize(),
                    yaxis_title='Rule',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Network Graph
            st.subheader("🕸️ Rules Network Graph")
            
            # Filter rules for visualization to keep it manageable
            graph_rules = rules.nlargest(min(20, len(rules)), 'lift')
            
            if len(graph_rules) > 1:
                G = nx.DiGraph()
                for _, rule in graph_rules.iterrows():
                    # Convert frozensets to strings for node labels
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    G.add_edge(antecedents, consequents, weight=rule['lift'])
                
                # Use a larger k value for spring layout to spread nodes out
                pos = nx.spring_layout(G, k=0.8/np.sqrt(len(G.nodes())), iterations=50)
                
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                node_x = []
                node_y = []
                node_text = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=True,
                        colorscale='YlOrRd',
                        size=20,
                        colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'
                        )
                    ))
                
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title=f'Association Rules Network (Top {len(graph_rules)} Rules)',
                                showlegend=False,
                                hovermode='closest',
                                height=600,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                             )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Network graph not shown as there are few or no rules to visualize.")
            
            # Download
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)
            with col1:
                csv = convert_df_to_csv(display_rules)
                st.download_button("📥 Download Rules (CSV)", csv, 
                                 "association_rules.csv", "text/csv")
            with col2:
                excel = convert_df_to_excel(display_rules)
                st.download_button("📥 Download Rules (Excel)", excel,
                                 "association_rules.xlsx", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================================================================
# TAB 2: CLASSIFICATION
# =============================================================================

def classification_tab(df):
    st.header("🎯 Classification Analysis")
    st.markdown("Predict customer behavior and segment classification")
    
    # Check for target columns
    required_cols = ['will_subscribe', 'willingness_to_pay_class']
    if not all(col in df.columns for col in required_cols):
        st.error("⚠️ The DataFrame is missing required target columns for Classification ('will_subscribe' or 'willingness_to_pay_class'). Please use the Sample Data or upload a file with these columns.")
        return
        
    # Target selection
    target_type = st.radio("Classification Type", 
                          ["Binary (Subscribe/Not Subscribe)", 
                           "Multi-Class (Willingness to Pay - 5 levels)"])
    
    if target_type.startswith("Binary"):
        target_col = 'will_subscribe'
        target_names = ['Not Subscribe', 'Subscribe']
    else:
        target_col = 'willingness_to_pay_class'
        target_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    available_models = ['Logistic Regression', 'Random Forest', 'SVM']
    if XGBClassifier:
        available_models.append('XGBoost')
        
    models_to_train = st.multiselect(
        "Select Models to Compare",
        available_models,
        default=['Logistic Regression', 'Random Forest']
    )
    
    if not models_to_train:
        st.warning("Please select at least one model")
        return
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 25) / 100
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Preprocess data
            df_processed, label_encoders = preprocess_data(df)
            
            # Prepare features and target
            feature_cols = get_feature_columns(df_processed)
            X = df_processed[feature_cols]
            y = df_processed[target_col]
            
            # Encode target if multi-class
            if target_type.startswith("Multi"):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                # Filter target names to only include present classes
                present_classes = sorted(np.unique(y))
                target_names = [target_names[i] for i in present_classes]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            results = {}
            
            for model_name in models_to_train:
                model = None
                if model_name == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_name == 'Random Forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name == 'SVM':
                    model = SVC(probability=True, random_state=42)
                elif model_name == 'XGBoost' and XGBClassifier:
                    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                
                if model is None:
                    continue

                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                # Need predict_proba for ROC and proper metric calculation
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            
            if not results:
                st.warning("No models were successfully trained.")
                return

            # Display results
            st.success("✅ Models trained successfully!")
            
            # Metrics comparison
            st.subheader("📊 Model Performance Comparison")
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [r['accuracy'] for r in results.values()],
                'Precision': [r['precision'] for r in results.values()],
                'Recall': [r['recall'] for r in results.values()],
                'F1-Score': [r['f1'] for r in results.values()]
            }).sort_values('Accuracy', ascending=False)
            
            st.dataframe(
                metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score']),
                use_container_width=True
            )
            
            # Best model
            best_model_name = metrics_df.iloc[0]['Model']
            best_result = results[best_model_name]
            
            st.info(f"🏆 Best Model: **{best_model_name}** (Accuracy: {best_result['accuracy']:.4f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                st.subheader("🔢 Confusion Matrix")
                cm = pd.crosstab(y_test, best_result['y_pred'], rownames=['Actual'], colnames=['Predicted'])
                
                # Use class labels from target_names
                if target_type.startswith("Multi"):
                    cm.columns = target_names
                    cm.index = target_names
                elif target_type.startswith("Binary"):
                    cm.columns = target_names
                    cm.index = target_names
                    
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=cm.columns,
                              y=cm.index,
                              color_continuous_scale='Blues',
                              text_auto=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Metrics comparison chart
                st.subheader("📈 Metrics Comparison")
                fig = go.Figure()
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                for model_name in results.keys():
                    values = [results[model_name]['accuracy'], 
                            results[model_name]['precision'],
                            results[model_name]['recall'],
                            results[model_name]['f1']]
                    fig.add_trace(go.Bar(name=model_name, x=metrics, y=values))
                
                fig.update_layout(barmode='group', height=500,
                                title='Model Metrics Comparison')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            importance_model_name = next((m for m in ['Random Forest', 'XGBoost'] if m in results), None)
            if importance_model_name:
                st.subheader(f"🔍 Feature Importance ({importance_model_name})")
                model = results[importance_model_name]['model']
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h', title='Top 10 Most Important Features')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve (for binary classification)
            if target_type.startswith("Binary"):
                st.subheader("📉 ROC Curves")
                fig = go.Figure()
                
                for model_name, result in results.items():
                    # Check if y_pred_proba has two columns (required for binary ROC)
                    if result['y_pred_proba'].shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'][:, 1])
                        roc_auc = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                               name=f'{model_name} (AUC = {roc_auc:.3f})'))
                
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                       name='Random Classifier', line=dict(dash='dash')))
                fig.update_layout(
                    title='ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            # Ensure target_names matches the number of unique classes in y_test
            unique_classes = np.unique(y_test)
            final_target_names = [target_names[c] for c in unique_classes]
            
            from sklearn.metrics import classification_report
            report = classification_report(y_test, best_result['y_pred'], 
                                         target_names=final_target_names,
                                         output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']),
                        use_container_width=True)
            
            # Download predictions
            st.subheader("💾 Download Results")
            predictions_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': best_result['y_pred'],
                'Correct': y_test == best_result['y_pred']
            })
            
            col1, col2 = st.columns(2)
            with col1:
                csv = convert_df_to_csv(predictions_df)
                st.download_button("📥 Download Predictions (CSV)", csv,
                                 "predictions.csv", "text/csv")
            with col2:
                csv_metrics = convert_df_to_csv(metrics_df)
                st.download_button("📥 Download Metrics (CSV)", csv_metrics,
                                 "model_metrics.csv", "text/csv")

# =============================================================================
# TAB 3: CLUSTERING
# =============================================================================

def clustering_tab(df):
    st.header("👥 Customer Clustering Analysis")
    st.markdown("Segment customers into distinct groups")
    
    # Preprocess data once for feature selection
    df_processed, _ = preprocess_data(df)

    # Feature selection
    st.subheader("📊 Select Features for Clustering")
    
    # Use encoded features for numeric properties where possible
    feature_options = [
        'age_group', 'gender', 'location', 'income', 'monthly_spending',
        'sustainability', 'artisan_support', 'shopping_style',
        'categories_interested', 'features_wanted'
    ]
    
    selected_features = st.multiselect(
        "Select features",
        feature_options,
        default=['income', 'monthly_spending', 'sustainability', 'categories_interested']
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features")
        return
    
    # Algorithm selection
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Clustering Algorithm", 
                                ['K-Means', 'Hierarchical', 'DBSCAN'])
    with col2:
        if algorithm != 'DBSCAN':
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        else:
            eps = st.slider("DBSCAN Eps", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 5)
    
    if st.button("🔬 Perform Clustering", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            
            # Select features (using encoded/numeric columns)
            feature_cols = [f'{f}_encoded' if f in df_processed.columns and f not in ['categories_interested', 'features_wanted']
                          else f for f in selected_features]
            
            # Filter df_processed to ensure all columns exist
            feature_cols = [col for col in feature_cols if col in df_processed.columns]
            
            if not feature_cols:
                st.error("Selected features could not be found or encoded.")
                return

            X = df_processed[feature_cols]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            labels = []
            n_clusters = 0
            
            if algorithm == 'K-Means':
                model = KMeans(n_clusters=n_clusters if 'n_clusters' in locals() else 4, 
                               random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif algorithm == 'Hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters if 'n_clusters' in locals() else 4)
                labels = model.fit_predict(X_scaled)
            else:  # DBSCAN
                model = DBSCAN(eps=eps if 'eps' in locals() else 0.5, 
                              min_samples=min_samples if 'min_samples' in locals() else 5)
                labels = model.fit_predict(X_scaled)
            
            # Determine actual number of clusters
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # Add labels to dataframe
            df_processed['Cluster'] = labels
            
            # Metrics
            if n_clusters > 1:
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
            else:
                silhouette = 0
                davies_bouldin = 0
                st.warning("Found 1 or 0 meaningful clusters (or only noise). Metrics are 0.")

            st.success(f"✅ Clustering complete! Found {n_clusters} clusters")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", n_clusters)
            with col2:
                st.metric("Silhouette Score", f"{silhouette:.4f}",
                         help="Higher is better (range: -1 to 1)")
            with col3:
                st.metric("Davies-Bouldin Index", f"{davies_bouldin:.4f}",
                         help="Lower is better")
            
            # Cluster sizes
            st.subheader("📊 Cluster Distribution")
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[f'Cluster {i}' if i != -1 else 'Noise' for i in cluster_sizes.index],
                      y=cluster_sizes.values,
                      marker_color=px.colors.qualitative.Set3[:len(cluster_sizes)])
            ])
            fig.update_layout(title='Number of Customers per Cluster', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster profiles
            st.subheader("👥 Cluster Personas")
            
            # Calculate cluster statistics
            cluster_profiles = []
            for cluster_id in sorted(unique_labels):
                if cluster_id == -1:
                    continue
                
                cluster_data = df[df_processed['Cluster'] == cluster_id]
                profile = {
                    'Cluster': f'Cluster {cluster_id}',
                    'Size': len(cluster_data),
                    'Size %': f"{len(cluster_data)/len(df)*100:.1f}%"
                }
                
                # Add feature statistics (using original, un-encoded columns)
                for feature in selected_features:
                    if feature in ['categories_interested', 'features_wanted']:
                        profile[feature] = f"{cluster_data[feature].mean():.1f}"
                    else:
                        mode_val = cluster_data[feature].mode()
                        profile[feature] = mode_val[0] if not mode_val.empty else 'N/A'
                
                cluster_profiles.append(profile)
            
            profiles_df = pd.DataFrame(cluster_profiles)
            st.dataframe(profiles_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Elbow curve (for K-Means)
                if algorithm == 'K-Means':
                    st.subheader("📉 Elbow Curve (K-Means)")
                    inertias = []
                    K_range = range(2, 11)
                    
                    # Ensure X_scaled has enough samples/features
                    if X_scaled.shape[0] >= 11: 
                        for k in K_range:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))
                        fig.add_vline(x=n_clusters if 'n_clusters' in locals() else 4, line_dash="dash", line_color="red",
                                    annotation_text=f"Selected: {n_clusters if 'n_clusters' in locals() else 4}")
                        fig.update_layout(
                            title='Elbow Method',
                            xaxis_title='Number of Clusters (K)',
                            yaxis_title='Inertia',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data points to compute a meaningful Elbow Curve (K-Means).")

                # Silhouette scores
                if algorithm != 'DBSCAN' and X_scaled.shape[0] >= 11: # Also avoid for DBSCAN
                    st.subheader("📊 Silhouette Analysis")
                    silhouette_scores = []
                    K_range = range(2, 11)
                    for k in K_range:
                        if algorithm == 'K-Means':
                            model_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                        else:
                            model_temp = AgglomerativeClustering(n_clusters=k)
                        labels_temp = model_temp.fit_predict(X_scaled)
                        
                        # Only calculate if > 1 unique label is found
                        if len(set(labels_temp)) > 1:
                            score = silhouette_score(X_scaled, labels_temp)
                            silhouette_scores.append(score)
                        else:
                            silhouette_scores.append(0) # or NaN
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, 
                                           mode='lines+markers'))
                    fig.add_vline(x=n_clusters if 'n_clusters' in locals() else 4, line_dash="dash", line_color="red",
                                annotation_text=f"Selected: {n_clusters if 'n_clusters' in locals() else 4}")
                    fig.update_layout(
                        title='Silhouette Score by Number of Clusters',
                        xaxis_title='Number of Clusters (K)',
                        yaxis_title='Silhouette Score',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif X_scaled.shape[0] < 11:
                     st.info("Not enough data points for detailed Silhouette Analysis.")
            
            with col2:
                # PCA visualization
                st.subheader("🎨 PCA Visualization")
                
                if X_scaled.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
                    })
                    
                    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                   title=f'Clusters in 2D PCA Space<br>Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("At least 2 features are required for PCA Visualization.")
                
                # 3D PCA (if enough components)
                if X_scaled.shape[1] >= 3:
                    st.subheader("🎯 3D PCA Visualization")
                    pca_3d = PCA(n_components=3)
                    X_pca_3d = pca_3d.fit_transform(X_scaled)
                    
                    pca_3d_df = pd.DataFrame({
                        'PC1': X_pca_3d[:, 0],
                        'PC2': X_pca_3d[:, 1],
                        'PC3': X_pca_3d[:, 2],
                        'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
                    })
                    
                    fig = px.scatter_3d(pca_3d_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                                      title='3D PCA Visualization',
                                      color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)

            # Download results
            st.subheader("💾 Download Results")
            results_df = df.copy()
            results_df['Cluster'] = labels
            
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = convert_df_to_csv(results_df)
                st.download_button("📥 Download Cluster Assignments", csv,
                                 "cluster_assignments.csv", "text/csv")
            with col2:
                csv_profiles = convert_df_to_csv(profiles_df)
                st.download_button("📥 Download Cluster Profiles", csv_profiles,
                                 "cluster_profiles.csv", "text/csv")
            with col3:
                excel = convert_df_to_excel(results_df)
                st.download_button("📥 Download Full Results (Excel)", excel,
                                 "clustering_results.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================================================================
# TAB 4: REGRESSION
# =============================================================================

def regression_tab(df):
    st.header("📈 Regression Analysis")
    st.markdown("Predict customer willingness to pay (numeric)")
    
    # Check for target column
    if 'willingness_to_pay_numeric' not in df.columns:
        st.error("⚠️ The DataFrame is missing the required target column **'willingness_to_pay_numeric'** for Regression Analysis. Please use the Sample Data or upload a file with this column.")
        return
        
    # Model selection
    st.subheader("🤖 Model Selection")
    
    available_models = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting']
    if XGBRegressor:
        available_models.append('XGBoost')

    models_to_train = st.multiselect(
        "Select Models to Compare",
        available_models,
        default=['Linear Regression', 'Random Forest']
    )
    
    if not models_to_train:
        st.warning("Please select at least one model")
        return
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 25) / 100
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training regression models..."):
            # Preprocess data
            df_processed, _ = preprocess_data(df)
            
            # Prepare features and target
            feature_cols = get_feature_columns(df_processed)
            X = df_processed[feature_cols]
            y = df_processed['willingness_to_pay_numeric']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            results = {}
            
            for model_name in models_to_train:
                model = None
                if model_name == 'Linear Regression':
                    model = LinearRegression()
                elif model_name == 'Ridge':
                    model = Ridge(random_state=42)
                elif model_name == 'Lasso':
                    model = Lasso(random_state=42)
                elif model_name == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name == 'Gradient Boosting':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_name == 'XGBoost' and XGBRegressor:
                    model = XGBRegressor(random_state=42)
                
                if model is None:
                    continue

                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae = mean_absolute_error(y_test, y_pred_test)
                
                # Adjusted R² - Check if n > p + 1
                n = len(y_test)
                p = X_test.shape[1]
                if n > p + 1:
                    adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
                else:
                    adjusted_r2 = np.nan
                
                results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'adjusted_r2': adjusted_r2,
                    'rmse': rmse,
                    'mae': mae,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                }
            
            if not results:
                st.warning("No models were successfully trained.")
                return

            # Display results
            st.success("✅ Models trained successfully!")
            
            # Metrics comparison
            st.subheader("📊 Model Performance Comparison")
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Train R²': [r['train_r2'] for r in results.values()],
                'Test R²': [r['test_r2'] for r in results.values()],
                'Adjusted R²': [r['adjusted_r2'] for r in results.values()],
                'RMSE': [r['rmse'] for r in results.values()],
                'MAE': [r['mae'] for r in results.values()]
            }).sort_values('Test R²', ascending=False)
            
            st.dataframe(
                metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Test R²', 'Adjusted R²'])
                                .background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE'], axis=0, vmin=metrics_df[['RMSE', 'MAE']].min().min(), vmax=metrics_df[['RMSE', 'MAE']].max().max()),
                use_container_width=True
            )
            
            # Best model
            best_model_name = metrics_df.iloc[0]['Model']
            best_result = results[best_model_name]
            
            st.info(f"🏆 Best Model: **{best_model_name}** (Test R²: {best_result['test_r2']:.4f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                st.subheader("🎯 Actual vs Predicted")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=best_result['y_pred_test'],
                                       mode='markers', name='Predictions',
                                       marker=dict(color='blue', opacity=0.6)))
                # Adjust range for 45-degree line
                min_val = min(y_test.min(), best_result['y_pred_test'].min())
                max_val = max(y_test.max(), best_result['y_pred_test'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val],
                                       y=[min_val, max_val],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(
                    title=f'{best_model_name} - Test Set',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residual plot
                st.subheader("📉 Residual Analysis")
                residuals = y_test - best_result['y_pred_test']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=best_result['y_pred_test'], y=residuals,
                                       mode='markers',
                                       marker=dict(color='red', opacity=0.6)))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title='Residual Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison chart
            st.subheader("📊 R² Metrics Comparison Across Models")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Test R²', x=list(results.keys()),
                               y=[r['test_r2'] for r in results.values()],
                               marker_color='lightblue'))
            # Filter out NaN Adjusted R² values for plotting
            valid_adj_r2 = [r['adjusted_r2'] if not pd.isna(r['adjusted_r2']) else 0 for r in results.values()]

            fig.add_trace(go.Bar(name='Adjusted R²', x=list(results.keys()),
                               y=valid_adj_r2,
                               marker_color='lightgreen'))
            fig.update_layout(barmode='group', height=400,
                            title='R² Comparison')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (for tree-based models)
            importance_model_name = next((m for m in ['Random Forest', 'Gradient Boosting', 'XGBoost'] if m in results), None)
            if importance_model_name:
                st.subheader(f"🔍 Feature Importance ({importance_model_name})")
                model = best_result['model']
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h', title='Top 15 Most Important Features')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Residual distribution
            st.subheader("📊 Residual Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(residuals, nbins=30, title='Histogram of Residuals')
                fig.update_layout(xaxis_title='Residuals', yaxis_title='Frequency', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Q-Q Plot
                fig = go.Figure()
                stats_result = stats.probplot(residuals, dist="norm")
                fig.add_trace(go.Scatter(x=stats_result[0][0], y=stats_result[0][1],
                                       mode='markers', name='Data'))
                fig.add_trace(go.Scatter(x=stats_result[0][0],
                                       y=stats_result[1][0] * stats_result[0][0] + stats_result[1][1],
                                       mode='lines', name='Normal Distribution',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(title='Q-Q Plot', xaxis_title='Theoretical Quantiles',
                                yaxis_title='Sample Quantiles', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            predictions_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': best_result['y_pred_test'],
                'Residual': residuals.values # Use .values to handle Series/Array
            })
            
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = convert_df_to_csv(predictions_df)
                st.download_button("📥 Download Predictions", csv,
                                 "regression_predictions.csv", "text/csv")
            with col2:
                csv_metrics = convert_df_to_csv(metrics_df)
                st.download_button("📥 Download Metrics", csv_metrics,
                                 "regression_model_metrics.csv", "text/csv")
            with col3:
                if importance_model_name:
                    csv_importance = convert_df_to_csv(feature_importance)
                    st.download_button("📥 Download Feature Importance", csv_importance,
                                     "regression_feature_importance.csv", "text/csv")

# =============================================================================
# TAB 5: DYNAMIC PRICING
# =============================================================================

def dynamic_pricing_tab(df):
    st.header("💰 Dynamic Pricing Engine")
    st.markdown("Get personalized price recommendations based on customer profile")
    
    # Check for target column
    required_cols = ['willingness_to_pay_numeric']
    if not all(col in df.columns for col in required_cols):
        st.error("⚠️ The DataFrame is missing required columns for Dynamic Pricing. Please use the Sample Data or upload a file with the 'willingness_to_pay_numeric' column.")
        return
    
    st.info("ℹ️ This tool uses a Random Forest regression model trained on the data to predict willingness to pay and suggests optimal pricing.")
    
    # Train a quick model for pricing
    with st.spinner("Loading pricing model..."):
        try:
            df_processed, label_encoders = preprocess_data(df)
            feature_cols = get_feature_columns(df_processed)
            X = df_processed[feature_cols]
            y = df_processed['willingness_to_pay_numeric']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest (best performing model typically)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
        except Exception as e:
            st.error(f"Failed to train the pricing model. Ensure data integrity. Error: {e}")
            return
            
    st.success("✅ Pricing model loaded!")
    
    # Customer input form
    st.subheader("👤 Customer Profile Input")
    
    # Get available categories from the original data columns for selectboxes
    age_groups = df['age_group'].unique()
    genders = df['gender'].unique()
    locations = df['location'].unique()
    income_levels = df['income'].unique()
    monthly_spendings = df['monthly_spending'].unique()
    shopping_styles = df['shopping_style'].unique()
    sustainability_levels = df['sustainability'].unique()
    artisan_supports = df['artisan_support'].unique()
    
    # Ensure all are sorted and in list form for clean display
    for var in [age_groups, genders, locations, income_levels, monthly_spendings, shopping_styles, sustainability_levels, artisan_supports]:
        if 'numpy' in str(type(var)): # Check for numpy array
             var.sort()
        elif hasattr(var, 'sort'): # Check for list
            var.sort()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_group = st.selectbox("Age Group", age_groups)
        gender = st.selectbox("Gender", genders)
        location = st.selectbox("Location", locations)
    
    with col2:
        income = st.selectbox("Income Level", income_levels)
        monthly_spending = st.selectbox("Current Monthly Spending", monthly_spendings)
        shopping_style = st.selectbox("Shopping Style", shopping_styles)
    
    with col3:
        sustainability = st.selectbox("Sustainability Consciousness", sustainability_levels)
        artisan_support = st.selectbox("Artisan Support Interest", artisan_supports)
        categories_interested = st.slider("Number of Categories Interested", 1, 6, 3)
        features_wanted = st.slider("Number of Features Wanted", 1, 8, 4)
    
    # Pricing strategy options
    st.subheader("🎯 Pricing Strategy")
    col1, col2 = st.columns(2)
    
    with col1:
        base_price = st.number_input("Base Product Price (₹)", 1000, 50000, 5000, 500)
        markup_strategy = st.selectbox("Markup Strategy",
                                      ['Conservative (10-20%)', 'Moderate (20-35%)', 
                                       'Aggressive (35-50%)', 'Premium (50-75%)'])
    
    with col2:
        demand_factor = st.slider("Demand Factor", 0.5, 2.0, 1.0, 0.1,
                                 help="Adjust based on inventory and demand (>1 for high demand)")
        loyalty_discount = st.slider("Loyalty Discount (%)", 0, 30, 0, 5,
                                    help="Discount for returning customers")
    
    if st.button("💡 Calculate Optimal Price", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'age_group': [age_group],
            'gender': [gender],
            'location': [location],
            'income': [income],
            'monthly_spending': [monthly_spending],
            'sustainability': [sustainability],
            'artisan_support': [artisan_support],
            'shopping_style': [shopping_style],
            'categories_interested': [categories_interested],
            'features_wanted': [features_wanted]
        })
        
        # Encode input
        # Note: This relies on the global `label_encoders` from the initial model training run.
        # This is a safe assumption for this synthetic data app structure.
        
        try:
            for col in label_encoders.keys():
                if col in input_data.columns:
                    # Handle unseen labels by setting to a default value (e.g., 0)
                    try:
                        input_data[f'{col}_encoded'] = label_encoders[col].transform(input_data[col])
                    except ValueError:
                        st.warning(f"Unseen label in column '{col}'. Setting encoded value to 0.")
                        input_data[f'{col}_encoded'] = 0
            
            # Prepare feature vector
            input_features = []
            for col in feature_cols:
                if col in input_data.columns:
                    input_features.append(input_data[col].values[0])
                elif col.replace('_encoded', '') in input_data.columns:
                    # For original non-encoded numeric features
                    input_features.append(input_data[col.replace('_encoded', '')].values[0])
                else:
                    st.error(f"Missing required feature column: {col}")
                    return

            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # Predict willingness to pay
            predicted_wtp_raw = model.predict(input_scaled)[0]
            
            # Calculate confidence interval
            predictions = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
            ci_lower_raw = np.percentile(predictions, 5)
            ci_upper_raw = np.percentile(predictions, 95)
            
        except Exception as e:
            st.error(f"Error during prediction or encoding. Check input values. Details: {e}")
            return

        # Simplified WTP to Max Price Mapping (Based on prediction scale)
        # The WTP numeric scale is approx 0 to 7 (synthetic data generation)
        # We need a business rule to link this score to a max price.
        # For demonstration, we'll use a linear scale and clamp.
        
        # WTP score (0-7) to Max Price (e.g., 2000 to 18000)
        # Max WTP formula: Max_Price = Min_Base + (WTP_Score / Max_Score) * Price_Range
        MIN_WTP_PRICE = 1500
        MAX_WTP_PRICE = 20000
        MAX_WTP_SCORE = 7 
        
        def score_to_price(score):
            score = max(0, min(MAX_WTP_SCORE, score)) # Clamp score between 0 and 7
            price = MIN_WTP_PRICE + (score / MAX_WTP_SCORE) * (MAX_WTP_PRICE - MIN_WTP_PRICE)
            return round(price / 100) * 100 # Round to nearest 100 for clean pricing
        
        predicted_max_price = score_to_price(predicted_wtp_raw)
        ci_lower_price = score_to_price(ci_lower_raw)
        ci_upper_price = score_to_price(ci_upper_raw)
        
        # Apply markup strategy
        if 'Conservative' in markup_strategy:
            markup_range = (0.10, 0.20)
        elif 'Moderate' in markup_strategy:
            markup_range = (0.20, 0.35)
        elif 'Aggressive' in markup_strategy:
            markup_range = (0.35, 0.50)
        else:  # Premium
            markup_range = (0.50, 0.75)
        
        # Calculate recommended price
        optimal_price = base_price * (1 + np.mean(markup_range))
        
        # Adjust for demand
        optimal_price *= demand_factor
        
        # Apply loyalty discount
        final_price = optimal_price * (1 - loyalty_discount / 100)
        
        # FINAL CONSTRAINT: Ensure final price is BELOW the maximum predicted WTP
        # Use the mid-point of the confidence interval as a reference floor
        max_acceptable_price = predicted_max_price
        
        # If the calculated final price is much higher than WTP, reduce it.
        if final_price > max_acceptable_price:
            final_price = max_acceptable_price * 0.95 # Safety buffer 
            st.warning(f"Price capped at 95% of predicted Max WTP (₹{max_acceptable_price:,.0f}) to ensure conversion.")
        
        # Ensure it's not below the base price (for profit)
        final_price = max(final_price, base_price * 1.05) # Minimum 5% margin
        
        # Conversion Probability Estimate (Simplified)
        # P = 1 - (Price / Max WTP) -> Clamped between 0 and 1
        conversion_prob = max(0, min(1, 1 - (final_price / predicted_max_price)))
        
        # Display results
        st.success("✅ Pricing Calculation Complete!")
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Predicted Max WTP", f"₹{predicted_max_price:,.0f}",
                     help="Maximum price customer is likely to pay")
        with col2:
            st.metric("Recommended Price", f"₹{final_price:,.0f}",
                     delta=f"{((final_price - base_price) / base_price * 100):.1f}% Markup")
        with col3:
            st.metric("Profit Margin", f"{((final_price - base_price) / final_price * 100):.1f}%")
        with col4:
            st.metric("Conversion Probability", 
                     f"{conversion_prob * 100:.1f}%",
                     help="Estimated likelihood of customer purchase at this price point.")
        
        # Detailed analysis
        st.subheader("📊 Pricing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price range visualization
            st.markdown("**Price Positioning**")
            
            fig = go.Figure()
            
            # Add range (Lower CI, WTP, Upper CI)
            fig.add_trace(go.Scatter(
                x=[ci_lower_price, predicted_max_price, ci_upper_price],
                y=['WTP Range', 'WTP Range', 'WTP Range'],
                mode='markers',
                marker=dict(size=[15, 25, 15], color=['lightblue', 'blue', 'lightblue']),
                name='WTP 90% CI'
            ))
            
            # Add base price
            fig.add_trace(go.Scatter(
                x=[base_price],
                y=['Base Price'],
                mode='markers',
                marker=dict(size=20, color='orange', symbol='diamond'),
                name='Base Price'
            ))
            
            # Add recommended price
            fig.add_trace(go.Scatter(
                x=[final_price],
                y=['Recommended Price'],
                mode='markers',
                marker=dict(size=20, color='green', symbol='star'),
                name='Recommended Price'
            ))
            
            fig.update_layout(
                title='Price Positioning Relative to WTP',
                xaxis_title='Price (₹)',
                height=350,
                showlegend=True,
                yaxis=dict(categoryorder='array', categoryarray=['WTP Range', 'Base Price', 'Recommended Price'])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price sensitivity
            st.markdown("**Price Sensitivity Analysis**")
            
            price_points = np.linspace(base_price * 0.9, predicted_max_price * 1.1, 20)
            conversion_probs = [max(0, min(100, (1 - (p / predicted_max_price)) * 100)) 
                              for p in price_points]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_points, y=conversion_probs, mode='lines',
                                   line=dict(color='blue', width=2)))
            fig.add_vline(x=final_price, line_dash="dash", line_color="green",
                         annotation_text=f"Recommended: ₹{final_price:,.0f}",
                         annotation_position="top right")
            fig.update_layout(
                title='Estimated Conversion Probability vs Price',
                xaxis_title='Price (₹)',
                yaxis_title='Conversion Probability (%)',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Pricing Recommendations")
        
        recommendations = []
        
        margin = (final_price - base_price) / final_price * 100
        if margin < 15:
            recommendations.append(f"**Low Margin:** Current margin ({margin:.1f}%) is conservative. Consider increasing the markup, especially if demand is high.")
        elif margin > 40:
            recommendations.append(f"**High Margin:** Excellent profit margin ({margin:.1f}%). Ensure product quality and marketing justify this premium.")
        
        if demand_factor > 1.2:
            recommendations.append("**High Demand Factor:** Price premium is justified by high demand. Monitor inventory closely.")
        
        if loyalty_discount > 0:
            recommendations.append(f"**Loyalty Focus:** Offering a {loyalty_discount}% discount supports customer retention. Highlight this value.")
        
        if predicted_max_price > final_price * 1.3:
            recommendations.append("**Untapped Potential:** Predicted Max WTP is significantly higher. You could test a higher price point to maximize revenue.")
        
        if shopping_style == 'Luxury' and predicted_max_price < 10000:
             recommendations.append("**Profile Mismatch:** The 'Luxury' profile suggests higher price tolerance. The WTP prediction might be low for the current product category.")

        for rec in recommendations:
            st.markdown(f"* {rec}")
        
        # Strategy summary
        st.subheader("📋 Pricing Strategy Summary")
        
        strategy_data = {
            'Component': ['Base Price', 'Avg Markup %', 'Demand Factor', 'Loyalty Discount %', 
                         'Initial Calculated Price', 'Final Recommended Price', 'Max WTP', 'Safety Margin'],
            'Value': [
                f"₹{base_price:,.0f}",
                f"{np.mean(markup_range)*100:.1f}%",
                f"×{demand_factor:.1f}",
                f"-{loyalty_discount:.0f}%",
                f"₹{optimal_price:,.0f}",
                f"₹{final_price:,.0f}",
                f"₹{predicted_max_price:,.0f}",
                f"₹{(predicted_max_price - final_price):,.0f} ({((predicted_max_price - final_price) / predicted_max_price * 100):.1f}%)"
            ]
        }
        
        st.table(pd.DataFrame(strategy_data))
        
        # Download pricing report
        st.subheader("💾 Download Pricing Report")
        
        report_data = {
            'Customer Profile': {
                'Age Group': age_group, 'Gender': gender, 'Location': location,
                'Income': income, 'Monthly Spending': monthly_spending,
                'Shopping Style': shopping_style, 'Sustainability': sustainability,
                'Artisan Support': artisan_support
            },
            'Pricing Details': {
                'Base Price (₹)': base_price,
                'Recommended Price (₹)': round(final_price, 2),
                'Max WTP (₹)': predicted_max_price,
                'Profit Margin %': round(margin, 2),
                'Conversion Probability %': round(conversion_prob * 100, 2)
            }
        }
        
        report_df = pd.DataFrame([
            {'Category': cat, 'Metric': k, 'Value': v}
            for cat, items in report_data.items()
            for k, v in items.items()
        ])
        
        csv = convert_df_to_csv(report_df)
        st.download_button("📥 Download Pricing Report", csv,
                         "pricing_report.csv", "text/csv")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize session state for DataFrame if not present
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/shop.png", width=80)
        st.title("Shopizz Analytics")
        st.markdown("---")
        
        st.subheader("📊 Data Management")
        
        data_source = st.radio(
            "Data Source",
            ["Use Sample Data", "Upload CSV", "Load from GitHub"]
        )
        
        # Button flags to trigger data loading/generation
        data_loaded = False
        
        if data_source == "Use Sample Data":
            n_samples = st.slider("Number of samples", 100, 2000, 1000, 100)
            if st.button("Generate Sample Data", key="gen_data"):
                df = generate_synthetic_data(n_samples)
                st.session_state['df'] = df
                st.success(f"✅ Generated {len(df)} samples")
                data_loaded = True
        
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success(f"✅ Loaded {len(df)} rows")
                data_loaded = True
        
        else:  # GitHub
            github_url = st.text_input(
                "GitHub Raw URL",
                placeholder="https://raw.githubusercontent.com/..."
            )
            if st.button("Load from GitHub", key="load_github"):
                df = load_data_from_github(github_url)
                if df is not None:
                    st.session_state['df'] = df
                    st.success(f"✅ Loaded {len(df)} rows from GitHub")
                    data_loaded = True
        
        # Data summary
        df_current = st.session_state['df']
        if df_current is not None:
            st.markdown("---")
            st.subheader("📈 Data Summary")
            st.write(f"**Rows:** {len(df_current)}")
            st.write(f"**Columns:** {len(df_current.columns)}")
            
            with st.expander("View Data"):
                st.dataframe(df_current.head(10))
            
            with st.expander("View Columns"):
                st.dataframe(df_current.dtypes.rename('DataType'))

        st.markdown("---")
        st.markdown("### 📖 About")
        st.info(
            "Shopizz Analytics Dashboard provides comprehensive ML-powered "
            "insights for customer behavior analysis, segmentation, and pricing optimization."
        )
    
    # Main content
    st.title("🛍️ Shopizz.com Analytics Dashboard")
    st.markdown("Comprehensive ML-powered customer analytics platform")
    
    df = st.session_state['df']
    
    if df is None:
        st.warning("⚠️ Please load data from the sidebar to begin analysis (using 'Generate Sample Data' is recommended to ensure all required columns are present).")
        
        # Show welcome screen features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 🔗 Association Rules
                Discover shopping patterns and product associations
            """)
        
        with col2:
            st.markdown("""
                ### 🎯 Classification
                Predict customer subscription and behavior
            """)
        
        with col3:
            st.markdown("""
                ### 👥 Clustering
                Segment customers into personas
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                ### 📈 Regression
                Predict willingness to pay
            """)
        
        with col2:
            st.markdown("""
                ### 💰 Dynamic Pricing
                Optimize pricing strategy
            """)
        
        return
    
    # Create tabs
    tabs = st.tabs([
        "🔗 Association Rules",
        "🎯 Classification",
        "👥 Clustering",
        "📈 Regression",
        "💰 Dynamic Pricing"
    ])
    
    with tabs[0]:
        association_rules_tab(df)
    
    with tabs[1]:
        classification_tab(df)
    
    with tabs[2]:
        clustering_tab(df)
    
    with tabs[3]:
        regression_tab(df)
    
    with tabs[4]:
        dynamic_pricing_tab(df)

if __name__ == "__main__":
    main()

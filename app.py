"""
Shopizz.com Analytics Dashboard
A comprehensive ML analytics platform for customer behavior analysis
(Refactored to a multi-page app structure)
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
from datetime import datetime
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from scipy import stats

# Attempt to import XGBoost
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

# Custom CSS (from your friend's code)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION & LOADING FUNCTIONS
# ============================================================================

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
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_data_from_github(url):
    """Load data from GitHub raw URL"""
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Convert dataframe to Excel for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_data(df):
    """Preprocess data for ML models and store encoders"""
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
            
    # Also create a numeric version of monthly_spending for EDA/Dashboard
    if 'monthly_spending' in df_processed.columns:
        spending_map = {
            '<1K': 500,
            '1-3K': 2000,
            '3-5K': 4000,
            '5-10K': 7500,
            '>10K': 12500
        }
        df_processed['monthly_spending_numeric'] = df_processed['monthly_spending'].map(spending_map).fillna(0)
    
    return df_processed, label_encoders

def get_feature_columns(df):
    """Get encoded feature columns for modeling"""
    return [col for col in df.columns if col.endswith('_encoded')] + \
           [col for col in df.columns if col in ['categories_interested', 'features_wanted']]

# ============================================================================
# PAGE 1: HOME & DATA MANAGEMENT
# ============================================================================

def page_home():
    st.markdown('<h2 class="sub-header">🏠 Welcome to the Shopizz Analytics Suite</h2>', 
                unsafe_allow_html=True)
    
    # STATUS INDICATOR
    st.markdown("### 📊 Data Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.df is not None:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ No Data")
    
    with col2:
        if st.session_state.df_processed is not None:
            st.success("✅ Data Processed")
        else:
            st.warning("⚠️ Data Not Processed")
    
    with col3:
        if st.session_state.df is not None:
            st.info(f"📊 {len(st.session_state.df)} rows")
        else:
            st.info("📊 0 rows")
    
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("### 📁 STEP 1: Load Data")
    
    tab1, tab2, tab3 = st.tabs(["🎲 Generate Sample Data", "📤 Upload CSV", "🔗 Load from GitHub"])
    
    with tab1:
        st.markdown("**Generate synthetic customer data for testing**")
        n_samples = st.slider("Number of samples to generate", 100, 2000, 1000, 100)
        
        if st.button("🎲 Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                df = generate_synthetic_data(n_samples)
                st.session_state.df = df
                st.session_state.df_processed = None # Reset processed data
                st.session_state.label_encoders = None
                st.session_state.analysis_results = {}
                st.success(f"✅ Generated {n_samples} synthetic records!")
                st.rerun()

    with tab2:
        st.markdown("**Upload your customer data CSV file**")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_upload_main")
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df, error = load_data(uploaded_file)
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.df = df
                    st.session_state.df_processed = None
                    st.session_state.label_encoders = None
                    st.session_state.analysis_results = {}
                    st.success(f"✅ Data loaded successfully! {len(df)} rows")
                    st.rerun()

    with tab3:
        st.markdown("**Load data from a GitHub Raw URL**")
        github_url = st.text_input(
            "GitHub Raw URL",
            placeholder="https://raw.githubusercontent.com/..."
        )
        if st.button("🔗 Load from GitHub"):
            with st.spinner("Loading data from GitHub..."):
                df, error = load_data_from_github(github_url)
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.df = df
                    st.session_state.df_processed = None
                    st.session_state.label_encoders = None
                    st.session_state.analysis_results = {}
                    st.success(f"✅ Data loaded successfully! {len(df)} rows")
                    st.rerun()

    # Show data preview if loaded
    if st.session_state.df is not None:
        with st.expander("👀 Preview Loaded Data (Click to expand)"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
    # STEP 2: PROCESS DATA
    st.markdown("---")
    st.markdown("### 🔧 STEP 2: Process Data (REQUIRED for Analysis)")
    
    if st.session_state.df is not None:
        if st.session_state.df_processed is not None:
            st.success("✅ Data is already processed and ready for analysis!")
            with st.expander("📊 View Processed Data Summary"):
                st.dataframe(st.session_state.df_processed.head(5), use_container_width=True)
        else:
            st.info("⚠️ Click the button below to process your data before running any analysis")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔧 PROCESS DATA NOW", type="primary", use_container_width=True):
                    with st.spinner("Processing data and creating features..."):
                        try:
                            df_processed, label_encoders = process_data(st.session_state.df)
                            
                            st.session_state.df_processed = df_processed
                            st.session_state.label_encoders = label_encoders
                            
                            st.success("✅ Data processed successfully!")
                            st.balloons()
                            
                            new_cols = [c for c in df_processed.columns if c not in st.session_state.df.columns]
                            st.markdown(f"**Created {len(new_cols)} new features:**")
                            st.write(new_cols)
                            st.info("🎉 You can now use all analysis features from the sidebar!")

                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
                            st.error("Please check your data format and try again")
    else:
        st.warning("⚠️ Please load data first (Step 1 above)")


# ============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (New)
# ============================================================================

def page_eda():
    st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please load and process data from the 'Home' page first.")
        return
    
    df = st.session_state.df
    df_processed = st.session_state.df_processed
    
    tab1, tab2, tab3 = st.tabs(["📈 Statistical Summary", "📊 Distributions", "🔗 Correlations"])
    
    with tab1:
        st.markdown("### 📈 Numeric Data Summary")
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        st.dataframe(df_processed[numeric_cols].describe().T, use_container_width=True)
        
        st.markdown("### 📋 Categorical Data Summary")
        categorical_cols = df.select_dtypes(include='object').columns
        st.dataframe(df[categorical_cols].describe().T, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Variable Distributions")
        
        # Select variable
        all_cols = df.columns.tolist()
        selected_var = st.selectbox("Select variable to visualize", all_cols, key="dist_select")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if df[selected_var].dtype in ['int64', 'float64']:
                fig = px.histogram(df, x=selected_var, title=f'Distribution of {selected_var}',
                                   marginal='box', color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = df[selected_var].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                             title=f'Distribution of {selected_var}')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'monthly_spending_numeric' in df_processed.columns and selected_var in categorical_cols:
                st.markdown(f"### 💰 Avg. Monthly Spending by {selected_var}")
                avg_spending = df_processed.groupby(selected_var)['monthly_spending_numeric'].mean().sort_values()
                fig = px.bar(avg_spending, y=avg_spending.index, x=avg_spending.values,
                             title=f'Avg. Monthly Spending by {selected_var}',
                             labels={'y': selected_var, 'x': 'Avg. Spending ($)'},
                             orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a categorical variable to see its relation to spending.")

    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df_processed[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            color_continuous_scale='RdBu_r',
                            aspect="auto",
                            title="Correlation Matrix of Numeric Features")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric features for a correlation matrix.")

# ============================================================================
# PAGE 3: ASSOCIATION RULE MINING (FIXED)
# ============================================================================

def page_association_rules():
    st.markdown('<h2 class="sub-header">🔗 Association Rule Mining</h2>', 
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data from the 'Home' page first.")
        return
        
    df = st.session_state.df
    
    # Validation Check
    if 'cart_items' not in df.columns:
        st.error("⚠️ The DataFrame is missing the required column **'cart_items'**. This analysis cannot be run.")
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
            
            # --- START: FIX FOR JSON SERIALIZABLE ERROR ---
            # Create a copy for visualization and convert frozensets to strings
            rules_for_viz = rules.copy()
            rules_for_viz['antecedents'] = rules_for_viz['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_for_viz['consequents'] = rules_for_viz['consequents'].apply(lambda x: ', '.join(list(x)))
            # --- END: FIX ---
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequent Itemsets", len(frequent_itemsets))
            with col2:
                st.metric("Association Rules", len(rules))
            with col3:
                st.metric("Avg Lift", f"{rules['lift'].mean():.2f}")
            
            # Top rules (use original 'rules' df for logic, but display converted strings)
            st.markdown("### 📊 Top 10 Rules by Lift")
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
                st.markdown("### 📈 Support vs Confidence")
                
                # --- MODIFIED LINE: Use 'rules_for_viz' DataFrame ---
                fig = px.scatter(rules_for_viz, x='support', y='confidence', size='lift',
                               color='lift', hover_data=['antecedents', 'consequents'],
                               title='Association Rules',
                               labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Top Rules by Metric")
                metric_choice = st.selectbox("Select Metric", ['lift', 'confidence', 'support'])
                
                # Use the 'rules_for_viz' df here too for consistency
                top_n = rules_for_viz.nlargest(10, metric_choice)
                
                fig = go.Figure(go.Bar(
                    x=top_n[metric_choice],
                    y=top_n.index, # Use index or a rule identifier
                    orientation='h',
                    marker=dict(color=top_n[metric_choice], colorscale='Viridis'),
                    # Add hover text with the converted strings
                    text=top_n.apply(lambda row: f"IF {row['antecedents']} THEN {row['consequents']}", axis=1),
                    hoverinfo="text+x"
                ))
                fig.update_layout(
                    title=f'Top 10 Rules by {metric_choice.capitalize()}',
                    xaxis_title=metric_choice.capitalize(),
                    yaxis_title='Rule',
                    yaxis=dict(autorange="reversed"), # Show top rule at the top
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Store the user-friendly string version
            st.session_state.analysis_results['association_rules'] = display_rules

# ============================================================================
# PAGE 4: CLASSIFICATION ANALYSIS
# ============================================================================

def page_classification():
    st.markdown('<h2 class="sub-header">🎯 Classification Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please load and process data from the 'Home' page first.")
        return

    df_processed = st.session_state.df_processed
    
    # Validation Check
    required_cols = ['will_subscribe', 'willingness_to_pay_class']
    if not all(col in df_processed.columns for col in required_cols):
        st.error("⚠️ The DataFrame is missing required target columns ('will_subscribe' or 'willingness_to_pay_class').")
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
    st.markdown("### 🤖 Model Selection")
    
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
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 25, key="clf_test_size") / 100
    
    if st.button("🚀 Train Classification Models", type="primary"):
        with st.spinner("Training models..."):
            
            # Prepare features and target
            feature_cols = get_feature_columns(df_processed)
            X = df_processed[feature_cols]
            y = df_processed[target_col]
            
            if X.empty or X.shape[0] < 2:
                st.error("🚨 The feature matrix is empty. Check data processing.")
                return

            # Encode target if multi-class
            if target_type.startswith("Multi"):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
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

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            
            if not results:
                st.warning("No models were successfully trained.")
                return

            st.success("✅ Models trained successfully!")
            
            # Metrics comparison
            st.markdown("### 📊 Model Performance Comparison")
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
                st.markdown("### 🔢 Confusion Matrix")
                cm = confusion_matrix(y_test, best_result['y_pred'])
                
                fig = px.imshow(cm, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=target_names[:len(np.unique(y_test))],
                                y=target_names[:len(np.unique(y_test))],
                                color_continuous_scale='Blues',
                                text_auto=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Metrics Comparison")
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
            
            # Feature Importance
            importance_model_name = next((m for m in ['Random Forest', 'XGBoost'] if m in results), None)
            if importance_model_name:
                st.markdown(f"### 🔍 Feature Importance ({importance_model_name})")
                model = results[importance_model_name]['model']
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h', title='Top 10 Most Important Features')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Store results
            st.session_state.analysis_results['classification'] = metrics_df

# ============================================================================
# PAGE 5: CUSTOMER CLUSTERING
# ============================================================================

def page_clustering():
    st.markdown('<h2 class="sub-header">👥 Customer Clustering Analysis</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please load and process data from the 'Home' page first.")
        return

    df = st.session_state.df
    df_processed = st.session_state.df_processed
    
    # Feature selection
    st.markdown("### 📊 Select Features for Clustering")
    all_numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    # Exclude target variables from default selection
    features_to_exclude = ['willingness_to_pay_numeric', 'will_subscribe']
    default_features = [col for col in all_numeric_cols if col not in features_to_exclude]

    selected_features = st.multiselect(
        "Select features",
        all_numeric_cols,
        default=default_features[:min(5, len(default_features))]
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
            
            X = df_processed[selected_features].copy().fillna(0) # Simple fillna

            if X.empty or X.shape[0] < 2:
                st.error("🚨 The resulting feature matrix is empty.")
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            labels = []
            
            if algorithm == 'K-Means':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif algorithm == 'Hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
            else: # DBSCAN
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Add labels to dataframe
            df_processed['Cluster'] = labels
            
            # Metrics
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
            else:
                silhouette = 0
                davies_bouldin = 0
                st.warning("Found 1 or 0 meaningful clusters. Metrics are 0.")

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
            st.markdown("### 📊 Cluster Distribution")
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[f'Cluster {i}' if i != -1 else 'Noise' for i in cluster_sizes.index],
                       y=cluster_sizes.values,
                       marker_color=px.colors.qualitative.Set3[:len(cluster_sizes)])
            ])
            fig.update_layout(title='Number of Customers per Cluster', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA visualization
            st.markdown("### 🎨 PCA Visualization")
            
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
            
            # Store results
            st.session_state.analysis_results['cluster_assignments'] = df_processed[['customer_id', 'Cluster']]


# ============================================================================
# PAGE 6: REGRESSION ANALYSIS
# ============================================================================

def page_regression():
    st.markdown('<h2 class="sub-header">📈 Regression Analysis (Willingness to Pay)</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please load and process data from the 'Home' page first.")
        return

    df_processed = st.session_state.df_processed
    
    if 'willingness_to_pay_numeric' not in df_processed.columns:
        st.error("⚠️ 'willingness_to_pay_numeric' column not found. Processing may have failed.")
        return
        
    # Model selection
    st.markdown("### 🤖 Model Selection")
    
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
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 25, key="reg_test_size") / 100
    
    if st.button("🚀 Train Regression Models", type="primary"):
        with st.spinner("Training regression models..."):
            
            # Prepare features and target
            feature_cols = get_feature_columns(df_processed)
            X = df_processed[feature_cols].copy().fillna(0) # Simple fillna
            y = df_processed['willingness_to_pay_numeric'].copy().fillna(0)

            if X.empty or X.shape[0] < 2:
                st.error("🚨 The feature matrix is empty.")
                return
            
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

                model.fit(X_train_scaled, y_train)
                y_pred_test = model.predict(X_test_scaled)
                
                results[model_name] = {
                    'model': model,
                    'test_r2': r2_score(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'mae': mean_absolute_error(y_test, y_pred_test),
                    'y_pred_test': y_pred_test
                }
            
            if not results:
                st.warning("No models were successfully trained.")
                return

            st.success("✅ Models trained successfully!")
            
            # Metrics comparison
            st.markdown("### 📊 Model Performance Comparison")
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Test R²': [r['test_r2'] for r in results.values()],
                'RMSE': [r['rmse'] for r in results.values()],
                'MAE': [r['mae'] for r in results.values()]
            }).sort_values('Test R²', ascending=False)
            
            st.dataframe(
                metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Test R²'])
                                .background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE']),
                use_container_width=True
            )
            
            # Best model
            best_model_name = metrics_df.iloc[0]['Model']
            best_result = results[best_model_name]
            
            st.info(f"🏆 Best Model: **{best_model_name}** (Test R²: {best_result['test_r2']:.4f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Actual vs Predicted")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=best_result['y_pred_test'],
                                         mode='markers', name='Predictions',
                                         marker=dict(color='blue', opacity=0.6)))
                min_val = min(y_test.min(), best_result['y_pred_test'].min())
                max_val = max(y_test.max(), best_result['y_pred_test'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
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
                st.markdown("### 📉 Residual Analysis")
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
            
            # Feature importance
            importance_model_name = next((m for m in ['Random Forest', 'Gradient Boosting', 'XGBoost'] if m in results), None)
            if importance_model_name:
                st.markdown(f"### 🔍 Feature Importance ({importance_model_name})")
                model = results[importance_model_name]['model']
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h', title='Top 15 Most Important Features')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.analysis_results['feature_importance'] = feature_importance

            # --- IMPORTANT: Save model for Dynamic Pricing ---
            st.session_state.best_regression_model = best_result['model']
            st.session_state.regression_scaler = scaler
            st.session_state.regression_features = feature_cols
            st.session_state.analysis_results['regression'] = metrics_df
            st.success("✅ Best regression model saved for use in the 'Dynamic Pricing Engine'!")


# ============================================================================
# PAGE 7: DYNAMIC PRICING ENGINE
# ============================================================================

def page_dynamic_pricing():
    st.markdown('<h2 class="sub-header">💰 Dynamic Pricing Engine</h2>', 
                unsafe_allow_html=True)
    
    if 'best_regression_model' not in st.session_state:
        st.error("🚨 Please run a model on the 'Regression Analysis' page first!")
        st.info("The Dynamic Pricing Engine uses the best model from that page to make predictions.")
        return
        
    if st.session_state.label_encoders is None:
        st.error("🚨 Label encoders not found. Please re-process data on the 'Home' page.")
        return

    # Load the trained model and components
    model = st.session_state.best_regression_model
    scaler = st.session_state.regression_scaler
    feature_cols = st.session_state.regression_features
    label_encoders = st.session_state.label_encoders
    df = st.session_state.df # Get raw df for options

    st.success("✅ Pricing model loaded successfully (using the best model from Regression page).")
    
    # Customer input form
    st.markdown("### 👤 Customer Profile Input")
    
    # Get available categories dynamically
    age_groups = df['age_group'].unique()
    genders = df['gender'].unique()
    locations = df['location'].unique()
    income_levels = df['income'].unique()
    monthly_spendings = df['monthly_spending'].unique()
    shopping_styles = df['shopping_style'].unique()
    sustainability_levels = df['sustainability'].unique()
    artisan_supports = df['artisan_support'].unique()

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
    st.markdown("### 🎯 Pricing Strategy")
    col1, col2 = st.columns(2)
    
    with col1:
        base_price = st.number_input("Base Product Price ($)", 10.0, 500.0, 50.0, 5.0)
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
            'age_group': [age_group], 'gender': [gender], 'location': [location],
            'income': [income], 'monthly_spending': [monthly_spending],
            'sustainability': [sustainability], 'artisan_support': [artisan_support],
            'shopping_style': [shopping_style],
            'categories_interested': [categories_interested],
            'features_wanted': [features_wanted]
        })
        
        # Encode input
        try:
            for col, le in label_encoders.items():
                if col in input_data.columns:
                    val = input_data[col].values[0]
                    if val in le.classes_:
                        input_data[f'{col}_encoded'] = le.transform(input_data[col])
                    else:
                        # Handle unseen label - use the most frequent (0)
                        st.warning(f"Unseen label '{val}' for {col}. Using default.")
                        input_data[f'{col}_encoded'] = 0 
            
            # Prepare feature vector in correct order
            input_vector = input_data[feature_cols]
            input_scaled = scaler.transform(input_vector)
            
            # Predict willingness to pay
            predicted_wtp = model.predict(input_scaled)[0]
            
            # Simple conversion from normalized WTP back to a price
            # This logic assumes the original WTP scale (0-~3.5)
            # We will map this to a dollar amount, e.g., max 3.5 = $200
            # This is an estimation!
            predicted_max_price = max(10, predicted_wtp * 50) # Example scaling
        
        except Exception as e:
            st.error(f"Error during input processing or prediction. Details: {e}")
            return

        # Apply markup strategy
        if 'Conservative' in markup_strategy:
            markup_range = (0.10, 0.20)
        elif 'Moderate' in markup_strategy:
            markup_range = (0.20, 0.35)
        elif 'Aggressive' in markup_strategy:
            markup_range = (0.35, 0.50)
        else: # Premium
            markup_range = (0.50, 0.75)
        
        # Calculate recommended price
        optimal_price = base_price * (1 + np.mean(markup_range))
        optimal_price *= demand_factor
        final_price = optimal_price * (1 - loyalty_discount / 100)
        
        # Ensure price doesn't exceed willingness to pay
        final_price = min(final_price, predicted_max_price)
        
        # Display results
        st.success("✅ Pricing Calculation Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Max WTP", f"${predicted_max_price:,.2f}",
                      help="Estimated max price customer is willing to pay")
        with col2:
            st.metric("Base Price", f"${base_price:,.2f}")
        with col3:
            st.metric("Recommended Price", f"${final_price:,.2f}",
                      delta=f"{((final_price - base_price) / base_price * 100):.1f}% vs Base")

        st.markdown("### 📋 Pricing Strategy Summary")
        strategy_data = {
            'Component': ['Base Price', 'Markup', 'Demand Adjustment', 'Loyalty Discount', 
                          'Final Price', 'Max WTP', 'Safety Margin'],
            'Value': [
                f"${base_price:,.2f}",
                f"+{np.mean(markup_range)*100:.0f}%",
                f"×{demand_factor:.1f}",
                f"-{loyalty_discount:.0f}%",
                f"${final_price:,.2f}",
                f"${predicted_max_price:,.2f}",
                f"${(predicted_max_price - final_price):,.2f}"
            ]
        }
        st.table(pd.DataFrame(strategy_data))

# ============================================================================
# PAGE 8: BUSINESS DASHBOARD (New)
# ============================================================================

def page_dashboard():
    st.markdown('<h2 class="sub-header">📈 Business Dashboard</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please load and process data from the 'Home' page first.")
        return

    df = st.session_state.df
    df_processed = st.session_state.df_processed

    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        avg_spending = df_processed['monthly_spending_numeric'].mean()
        st.metric("Avg. Monthly Spending", f"${avg_spending:,.2f}")
    with col3:
        top_location = df['location'].mode()[0]
        st.metric("Top Location", top_location)
    with col4:
        top_age_group = df['age_group'].mode()[0]
        st.metric("Top Age Group", top_age_group)
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛍️ Spending by Shopping Style")
        spending_by_style = df_processed.groupby('shopping_style')['monthly_spending_numeric'].mean().sort_values()
        fig = px.bar(spending_by_style, x=spending_by_style.values, y=spending_by_style.index,
                     orientation='h', labels={'x': 'Avg. Monthly Spending ($)', 'y': 'Shopping Style'},
                     title="Avg. Monthly Spending by Shopping Style")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### 🌍 Spending by Location")
        spending_by_loc = df_processed.groupby('location')['monthly_spending_numeric'].mean().sort_values()
        fig = px.bar(spending_by_loc, x=spending_by_loc.values, y=spending_by_loc.index,
                     orientation='h', labels={'x': 'Avg. Monthly Spending ($)', 'y': 'Location'},
                     title="Avg. Monthly Spending by Location")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌳 Sustainability vs. Spending")
        fig = px.box(df_processed, x='sustainability', y='monthly_spending_numeric',
                     title="Monthly Spending by Sustainability Interest",
                     labels={'sustainability': 'Sustainability Interest', 'monthly_spending_numeric': 'Monthly Spending ($)'},
                     color='sustainability')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🤝 Artisan Support vs. Spending")
        fig = px.box(df_processed, x='artisan_support', y='monthly_spending_numeric',
                     title="Monthly Spending by Artisan Support Interest",
                     labels={'artisan_support': 'Artisan Support Interest', 'monthly_spending_numeric': 'Monthly Spending ($)'},
                     color='artisan_support')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 9: EXPORT REPORTS (New)
# ============================================================================

def page_export():
    st.markdown('<h2 class="sub-header">📥 Export Reports & Data</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded to export.")
        return
        
    st.markdown("### 💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export raw data
        csv_raw = convert_df_to_csv(st.session_state.df)
        st.download_button(
            label="📥 Download Raw Data (CSV)",
            data=csv_raw,
            file_name=f"shopizz_raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export processed data
        if st.session_state.df_processed is not None:
            csv_processed = convert_df_to_csv(st.session_state.df_processed)
            st.download_button(
                label="📥 Download Processed Data (CSV)",
                data=csv_processed,
                file_name=f"shopizz_processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Process data on the 'Home' page to enable this download.")

    st.markdown("---")
    st.markdown("### 📊 Download Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'regression' in st.session_state.analysis_results:
            csv_reg = convert_df_to_csv(st.session_state.analysis_results['regression'])
            st.download_button(
                label="📥 Download Regression Metrics (CSV)",
                data=csv_reg,
                file_name=f"regression_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Run Regression analysis to enable.")

    with col2:
        if 'classification' in st.session_state.analysis_results:
            csv_clf = convert_df_to_csv(st.session_state.analysis_results['classification'])
            st.download_button(
                label="📥 Download Classification Metrics (CSV)",
                data=csv_clf,
                file_name=f"classification_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Run Classification analysis to enable.")

    with col3:
        if 'association_rules' in st.session_state.analysis_results:
            csv_assoc = convert_df_to_csv(st.session_state.analysis_results['association_rules'])
            st.download_button(
                label="📥 Download Association Rules (CSV)",
                data=csv_assoc,
                file_name=f"association_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Run Association Rules to enable.")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-header">🛍️ Shopizz.com Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/shop.png", width=100)
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select Analysis Module:",
            ["🏠 Home & Data Management",
             "📊 Exploratory Data Analysis",
             "🔗 Association Rule Mining",
             "🎯 Classification Analysis",
             "👥 Customer Clustering",
             "📈 Regression Analysis (WTP)",
             "💰 Dynamic Pricing Engine",
             "📈 Business Dashboard",
             "📥 Export Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        
        if 'df' in st.session_state and st.session_state.df is not None:
            st.metric("Total Records", len(st.session_state.df))
            st.metric("Total Features", len(st.session_state.df.columns))
            if 'df_processed' in st.session_state and st.session_state.df_processed is not None:
                st.metric("Processed Features", len(st.session_state.df_processed.columns))
        else:
            st.info("Load data to see stats")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Route to selected page
    if page == "🏠 Home & Data Management":
        page_home()
    elif page == "📊 Exploratory Data Analysis":
        page_eda()
    elif page == "🔗 Association Rule Mining":
        page_association_rules()
    elif page == "🎯 Classification Analysis":
        page_classification()
    elif page == "👥 Customer Clustering":
        page_clustering()
    elif page == "📈 Regression Analysis (WTP)":
        page_regression()
    elif page == "💰 Dynamic Pricing Engine":
        page_dynamic_pricing()
    elif page == "📈 Business Dashboard":
        page_dashboard()
    elif page == "📥 Export Reports":
        page_export()

if __name__ == "__main__":
    main()

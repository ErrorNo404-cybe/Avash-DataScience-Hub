import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import ttest_ind, chi2_contingency, normaltest
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import hashlib
import json
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Avash's Data Science Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🧮🔬 Avash's Data Science Hub")
module = st.sidebar.selectbox(
    "Select Course Module",
    [
        "🏠 Home",
        "Data Mining",
        "Cloud & Big Data",
        "Intro to Data Science",
        "Privacy & Security",
        "Statistical Analysis",
        "Visual Storytelling"
    ]
)

def generate_quick_insights(df):
    """
    Generate automated insights based on dataframe structure.
    Returns a list of (title, insight_text, optional_chart_func) tuples.
    """
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() < 50]
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    insights.append((
        "Dataset Overview",
        f"Your dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.",
        None
    ))
    
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        insights.append((
            "Missing Data Alert",
            f"⚠️ There are **{missing_total} missing values** across the dataset. Consider imputation or removal.",
            None
        ))
    
    high_card_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() > 100]
    if high_card_cols:
        insights.append((
            "Potential ID/Text Columns",
            f"Columns like **{', '.join(high_card_cols[:3])}** have high uniqueness — likely IDs or free text. Exclude from grouping.",
            None
        ))
    
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        insights.append((
            f"Distribution of {col}",
            f"Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}",
            lambda: st.line_chart(df[col].dropna().reset_index(drop=True))
        ))
    
    for col in categorical_cols:
        top_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        top_pct = top_freq / len(df) * 100
        if top_pct > 80:
            insights.append((
                f"Imbalance in {col}",
                f"Top category represents **{top_pct:.1f}%** of data — consider rebalancing for modeling.",
                None
            ))
    
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        corr_unstack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).unstack().dropna()
        if not corr_unstack.empty:
            max_pair = corr_unstack.idxmax()
            max_val = corr_unstack.max()
            if max_val > 0.5:
                insights.append((
                    "Strong Correlation Detected",
                    f"**{max_pair[0]}** and **{max_pair[1]}** are highly correlated (r = {max_val:.2f}).",
                    None
                ))
    
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        metric_col = numeric_cols[0]
        df_temp = df[[date_col, metric_col]].dropna()
        if len(df_temp) > 10:
            df_temp = df_temp.set_index(date_col).sort_index()
            if df_temp.index.is_monotonic_increasing or df_temp.index.is_monotonic_decreasing:
                insights.append((
                    f"Trend in {metric_col}",
                    f"Time-series data detected. Consider analyzing seasonality or growth.",
                    lambda: st.line_chart(df_temp[metric_col])
                ))
    
    return insights

def clean_dataframe(df_original):
    """Interactive data cleaning UI that returns a cleaned DataFrame."""
    df = df_original.copy()
    st.subheader("🔧 Data Cleaning & Transformation")
    
    if st.checkbox("🗑️ Drop duplicate rows", value=False):
        original_len = len(df)
        df = df.drop_duplicates()
        st.info(f"Dropped {original_len - len(df)} duplicate rows.")
    
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        st.write("ValueHandling **Fill missing values**")
        fill_method = st.radio(
            "Strategy",
            ["Mean (numeric only)", "Median (numeric only)", "Mode (most frequent)", "Drop rows with missing values"],
            key="missing_fill"
        )
        if st.button("Apply Missing Value Fix"):
            if "Drop rows" in fill_method:
                original_len = len(df)
                df = df.dropna()
                st.success(f"Dropped {original_len - len(df)} rows with missing values.")
            else:
                for col in df.columns:
                    if df[col].isnull().any():
                        if fill_method.startswith("Mean") and df[col].dtype in [np.float64, np.int64]:
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif fill_method.startswith("Median") and df[col].dtype in [np.float64, np.int64]:
                            df[col].fillna(df[col].median(), inplace=True)
                        elif fill_method == "Mode (most frequent)":
                            mode_val = df[col].mode()
                            if not mode_val.empty:
                                df[col].fillna(mode_val[0], inplace=True)
                st.success("Missing values filled.")
    
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_candidates = [c for c in object_cols if 'date' in c.lower() or 'time' in c.lower()]
    if datetime_candidates:
        st.write("📅 **Convert to datetime**")
        date_col = st.selectbox("Column to convert", datetime_candidates, key="date_col")
        if st.button("Convert to DateTime"):
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                st.success(f"Converted {date_col} to datetime.")
            except Exception as e:
                st.error(f"Failed: {e}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.write("🧮 **Create new feature (A / B)**")
        col_a = st.selectbox("Numerator (A)", numeric_cols, key="num")
        col_b = st.selectbox("Denominator (B)", numeric_cols, key="denom")
        new_col_name = st.text_input("New column name", value=f"{col_a}_per_{col_b}")
        if st.button("Create Ratio Feature"):
            if col_b != col_a:
                df[new_col_name] = df[col_a] / df[col_b]
                st.success(f"Created new column: {new_col_name}")
            else:
                st.warning("Numerator and denominator must be different.")
    
    return df

def upload_file():
    uploaded_file = st.file_uploader(
        "📤 Upload your **CSV** or **Excel** file to begin analysis",
        type=["csv", "xlsx"],
        help="All processing is local — your data never leaves your browser."
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            if df.empty:
                st.error("Uploaded file is empty.")
                return None
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

if module == "🏠 Home":
    st.title("📈🤖 Avash's Data Science Hub")
    st.markdown("""
    
    🔒 **Your data stays in your browser** — nothing is stored or transmitted.  
    📂 Upload a dataset in any module to unlock the required analysis tools.
    
    """)
    st.info("💡 Tip: Use real-world datasets (sales, surveys, logs, transactions) for the best experience.")

elif module == "Data Mining":
    st.title("Data Mining — Clustering, Classification, Association Rules")
    st.markdown("""
    Upload data for:
    - **Clustering**: Group similar records (unsupervised)
    - **Classification**: Predict categorical labels (supervised)
    - **Association Rules**: Find item co-occurrences (e.g., market basket)
    """)
    
    df = upload_file()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(["Clustering", "Classification", "Association Rules"])
        
        with tab1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("Need ≥2 numeric columns for clustering.")
            else:
                cols = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])
                n_clusters = st.slider("Clusters", 2, min(10, len(df)-1), 3)
                if len(cols) >= 2:
                    data = df[cols].dropna()
                    if len(data) > n_clusters:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        data['Cluster'] = kmeans.fit_predict(data_scaled)
                        fig = px.scatter(data, x=cols[0], y=cols[1], color='Cluster')
                        st.plotly_chart(fig)
                    else:
                        st.warning("Not enough rows after dropping NaN.")
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 20]
            if not numeric_cols or not cat_cols:
                st.error("Need ≥1 numeric column and ≥1 categorical column (for target).")
            else:
                target = st.selectbox("Target (label)", cat_cols)
                features = st.multiselect("Features", numeric_cols, default=numeric_cols[:3])
                if features and target:
                    X = df[features].copy()
                    y = df[target].copy()
                    valid = y.notna() & X.notna().all(axis=1)
                    X, y = X[valid], y[valid]
                    if len(X) > 10:
                        le = LabelEncoder()
                        y_enc = le.fit_transform(y)
                        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
                        clf = RandomForestClassifier(n_estimators=50)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
                        st.text(classification_report(y_test, y_pred, target_names=le.classes_))
                    else:
                        st.error("Not enough valid data.")
        
        with tab3:
            st.markdown("💡 **Format**: Each row = transaction, columns = items (1/0 or True/False).")
            if df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
                st.info("Detected numeric transaction data. Converting to boolean.")
                df_bool = df.astype(bool)
                frequent_itemsets = apriori(df_bool, min_support=0.1, use_colnames=True)
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    if not rules.empty:
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                    else:
                        st.warning("No strong association rules found (try lowering thresholds).")
                else:
                    st.warning("No frequent itemsets found. Try a dataset with more co-occurrences.")
            else:
                st.error("Association rules require a **binary/boolean transaction matrix**.")

elif module == "Cloud & Big Data":
    st.title("Cloud & Big Data — Concepts & Architecture")
    st.markdown("This module provides informative discussion — no file upload needed.")

    st.subheader("☁️ Cloud Computing Models")
    st.markdown("""
    - **IaaS** (Infrastructure as a Service):  
      [AWS EC2](https://aws.amazon.com/ec2/) • 
      [Azure VMs](https://azure.microsoft.com/en-us/products/virtual-machines) • 
      [Google Compute Engine](https://cloud.google.com/compute)
    - **PaaS** (Platform as a Service):  
      [Google App Engine](https://cloud.google.com/appengine) • 
      [Azure App Service](https://azure.microsoft.com/en-us/products/app-service) • 
      [Heroku](https://www.heroku.com/)
    - **SaaS** (Software as a Service):  
      [Gmail](https://mail.google.com) • 
      [Microsoft 365](https://www.microsoft.com/microsoft-365) • 
      [Salesforce](https://www.salesforce.com)
    """)

    st.subheader("📊 Big Data Frameworks")
    st.markdown("""
    - **Batch Processing**:  
      [Apache Hadoop](https://hadoop.apache.org) • 
      [Apache Spark](https://spark.apache.org)
    - **Stream Processing**:  
      [Apache Kafka](https://kafka.apache.org) • 
      [Apache Flink](https://flink.apache.org)
    - **Cloud Data Lakes**:  
      [Amazon S3](https://aws.amazon.com/s3) • 
      [Azure Data Lake](https://azure.microsoft.com/en-us/products/storage/data-lake-storage) • 
      [Google Cloud Storage](https://cloud.google.com/storage)
    """)

    st.subheader("🏗️ Reference Architecture: Cloud Data Pipeline")
    st.markdown("Example: End-to-end analytics on AWS")

    st.image(
    "https://d1.awsstatic.com/onedam/marketing-channels/website/aws/en_US/architecture/approved/images/a2a2572d2662393ab3636f7af7194a81-guidance-architecture-diagram-data-lakes-on-aws-2900x1741.a9164027e037e63ba8df08f2d5372928f00334c7.png",
    caption="AWS Data Lake Architecture (Source: AWS Documentation)",
    use_column_width=True
)
    st.markdown("[View full diagram on AWS](https://aws.amazon.com/solutions/guidance/data-lakes-on-aws)")

    st.subheader("🔒 Data Security & Compliance")
    st.markdown("""
    - **At Rest**: [AES-256](https://aws.amazon.com/s3/features/security/) encryption
    - **In Transit**: [TLS 1.3](https://datatracker.ietf.org/doc/html/rfc8446)
    - **Compliance**:  
      [GDPR](https://gdpr-info.eu) • 
      [HIPAA](https://www.hhs.gov/hipaa) • 
      [SOC 2](https://www.aicpa.org)
    """)

    st.subheader("📦 Containers & Observability")
    st.markdown("""
    - **Containers**: [Docker](https://www.docker.com)
    - **Orchestration**: [Kubernetes](https://kubernetes.io)
    - **Monitoring**:  
      [Amazon CloudWatch](https://aws.amazon.com/cloudwatch) • 
      [Prometheus + Grafana](https://prometheus.io)
    """)

    st.info("""
💡 **This app runs on [Streamlit Cloud](https://streamlit.io/cloud) (free PaaS)**, but production systems require:

- **Virtual machines** for custom code:  
  [AWS EC2](https://aws.amazon.com/ec2/) • 
  [Azure VMs](https://azure.microsoft.com/en-us/products/virtual-machines) • 
  [Google Compute Engine](https://cloud.google.com/compute)

- **Object storage** for raw data:  
  [Amazon S3](https://aws.amazon.com/s3/) • 
  [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs) • 
  [Google Cloud Storage](https://cloud.google.com/storage)

- **CI/CD & Infrastructure-as-Code (IaC)**:  
  [Terraform (by HashiCorp)](https://www.terraform.io) • 
  [AWS CloudFormation](https://aws.amazon.com/cloudformation/) • 
  [Azure Resource Manager (ARM)](https://learn.microsoft.com/en-us/azure/azure-resource-manager/)

- **Security**:  
  - **Role-based access control (IAM)**:  
    [AWS IAM](https://aws.amazon.com/iam/) • 
    [Azure RBAC](https://learn.microsoft.com/en-us/azure/role-based-access-control/) • 
    [Google IAM](https://cloud.google.com/iam)  
  - **Encryption everywhere**:  
    [AWS KMS](https://aws.amazon.com/kms/) • 
    [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault) • 
    [Google Cloud KMS](https://cloud.google.com/kms)
""")

elif module == "Intro to Data Science":
    st.title("Data Science Project Lifecycle")
    st.markdown("Upload your dataset to explore, analyze, and generate insights.")
    
    df = upload_file()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("1. Problem Definition")
        question = st.text_area(
            "What question are you trying to answer?",
            placeholder="e.g., Which segment has highest revenue? What drives churn?"
        )
        
        st.subheader("2. Quick Insights (Auto-Generated)")
        if st.button("🔍 Generate Insights"):
            insights = generate_quick_insights(df)
            for title, text, chart_func in insights:
                with st.expander(f"💡 {title}"):
                    st.markdown(text)
                    if chart_func:
                        chart_func()
        
        st.subheader("3. Full Data Exploration")
        
        st.write("**Column Overview**")
        col_summary = pd.DataFrame({
            "Data Type": df.dtypes,
            "Unique Values": df.nunique(),
            "Missing (%)": (df.isnull().sum() / len(df) * 100).round(1),
            "Sample Value": df.iloc[0].values  # First row as example
        })
        
        st.dataframe(col_summary, use_container_width=True)
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.write("**Missing Values by Column**")
            fig_missing = px.bar(
                x=missing.index,
                y=missing.values,
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Values"
            )
            st.plotly_chart(fig_missing, use_container_width=True)
            
        else:
            st.success("✅ No missing values!")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            st.write("**Numeric Distributions** (Top 2)")
            cols_to_plot = numeric_cols[:2]
            for col in cols_to_plot:
                fig = px.histogram(df, x=col, nbins=30, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                if categorical_cols:
                    st.write("**Categorical Distributions** (Top 2)")
                    cols_to_plot = [c for c in categorical_cols if df[c].nunique() <= 20][:2]  
                    for col in cols_to_plot:
                        value_counts = df[col].value_counts().reset_index()
                        value_counts.columns = [col, "Count"]
                        fig = px.bar(value_counts, x=col, y="Count", title=f"{col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
        high_card = [c for c in categorical_cols if df[c].nunique() > 20]
        if high_card:
            st.warning(f"⚠️ High-cardinality text columns (not plotted): {', '.join(high_card[:3])}")

        if st.button("📊 Generate Full EDA Report"):
            try:
                from ydata_profiling import ProfileReport
                with st.spinner("Generating report... this may take a few seconds"):
                    profile = ProfileReport(df, title="Data Science Hub - EDA Report", explorative=True, dark_mode=True)  
                    
                    profile_html = profile.to_html()
                    st.components.v1.html(profile_html, height=800, scrolling=True)
                    
                    st.download_button("📥 Download Report (HTML)", profile_html, "eda_report.html", "text/html")
            except Exception as e:
                st.error(f"Could not generate report: {e}")
        
        st.subheader("3. Data Manipulation")
        with st.expander("🔧 Clean & Transform"):
            df_clean = clean_dataframe(df)
            
            st.write("**Preview of Cleaned Data**")
            st.dataframe(df_clean.head(), use_container_width=True)
            
            st.download_button("📥 Download Cleaned Data",
                           df_clean.to_csv(index=False),"cleaned_data.csv","text/csv")
        
        st.subheader("4. Modeling & Evaluation")
        st.markdown("Use **Statistical Analysis with Experimental Design** or **Data Mining** modules for modeling.")
        
        st.subheader("5. Deployment & Communication")
        st.markdown("Use (Visual Storytelling) to share insights.")

elif module == "Privacy & Security":
    st.title("Privacy, Security & Compliance")
    st.markdown("Upload data containing sensitive fields (e.g., PII) to apply protections.")
    
    df = upload_file()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("🌍 Global Privacy Regulations")
        st.markdown("""
        - **GDPR** (EU): Right to erasure, data portability
        - **CCPA** (California): Opt-out of data sale
        - **HIPAA** (US Health): PHI protection
        - **PIPEDA** (Canada): Consent-based collection
        """)
        
        st.subheader("🛡️ Data Protection Strategies")
        cols = st.multiselect("Select sensitive columns to protect", df.columns)
        if cols:
            strategy = st.radio("Anonymization Method", ["Hashing", "Generalization", "Suppression"])
            df_protected = df.copy()
            if strategy == "Hashing":
                for col in cols:
                    df_protected[col] = df_protected[col].astype(str).apply(
                        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
                    )
            elif strategy == "Generalization":
                for col in cols:
                    if df_protected[col].dtype in ['int64', 'float64']:
                        df_protected[col] = "[REDACTED]"
                    else:
                        df_protected[col] = df_protected[col].astype(str).apply(lambda x: x[0] + "***")
            else:  
                df_protected[cols] = "[REDACTED]"
            
            st.write("**Protected Data**")
            st.dataframe(df_protected.head())
            st.download_button("📥 Download Protected Data", df_protected.to_csv(index=False), "protected.csv")
        
        st.subheader("⚠️ Risk Assessment")
        if st.button("Perform Risk Assessment"):
            risks = []
            if any(col.lower() in ['email', 'ssn', 'phone', 'name', 'id'] for col in df.columns):
                risks.append("⚠️ Contains PII — high risk if exposed")
            if df.isnull().sum().sum() > 0:
                risks.append("⚠️ Missing data may lead to biased models")
            if not cols:
                risks.append("❗ No anonymization applied — raw data is sensitive")
            
            if risks:
                for r in risks:
                    st.warning(r)
            else:
                st.success("✅ Low-risk profile (based on metadata)")

elif module == "Statistical Analysis":
    st.title("Experimental Design & Inference")
    st.markdown("""
    Upload data for:
    - **Hypothesis Testing** (t-test, chi-square)
    - **Normality Checks**
    - **A/B Test Design Guidance**
    """)
    
    df = upload_file()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("1. Inferential Statistics")
        numeric_cols = df.select_dtypes(np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column for distribution test", numeric_cols)
            data = df[col].dropna()
            if len(data) > 3:
                _, p_norm = normaltest(data)
                st.write(f"**Normality (p-value)**: {p_norm:.4f} → {'Normal' if p_norm > 0.05 else 'Non-normal'}")
        
        st.subheader("2. Hypothesis Testing")
        test_type = st.radio("Test Type", ["Two-Group Comparison (t-test)", "Categorical Association (Chi-square)"])
        
        if test_type == "Two-Group Comparison (t-test)":
            group_col = st.selectbox("Group column", df.columns)
            metric_col = st.selectbox("Metric", numeric_cols)
            if group_col and metric_col:
                groups = df[group_col].dropna().unique()
                if len(groups) == 2:
                    g1 = df[df[group_col] == groups[0]][metric_col].dropna()
                    g2 = df[df[group_col] == groups[1]][metric_col].dropna()
                    if len(g1) > 5 and len(g2) > 5:
                        t, p = ttest_ind(g1, g2, equal_var=False)
                        st.write(f"**p-value**: {p:.4f} → {'Significant' if p < 0.05 else 'Not significant'}")
        
        elif test_type == "Categorical Association (Chi-square)":
            col1 = st.selectbox("Categorical variable 1", [c for c in df.columns if df[c].dtype == 'object'])
            col2 = st.selectbox("Categorical variable 2", [c for c in df.columns if df[c].dtype == 'object'])
            if col1 and col2:
                table = pd.crosstab(df[col1], df[col2])
                if table.size > 0:
                    chi2, p, _, _ = chi2_contingency(table)
                    st.write(f"**Chi-square p-value**: {p:.4f} → {'Associated' if p < 0.05 else 'Independent'}")
        
        st.subheader("3. Experimental Design Tips")
        st.markdown("""
        - **Randomization**: Assign users randomly to groups
        - **Control Group**: Always include a baseline
        - **Sample Size**: Use power analysis (aim for 80% power)
        - **Duration**: Run long enough to capture behavior cycles
        """)

elif module in ["Visual Storytelling", "Business Intelligence"]:
    st.title("Visual Storytelling & Business Intelligence")
    st.markdown("""
    Upload **business data** (sales, marketing, ops) to:
    - Identify trends and outliers
    - Discover business opportunities
    - Formulate growth strategies
    - Build executive dashboards
    """)
    
    df = upload_file()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("1. Strategic Data Exploration")
        numeric = df.select_dtypes(np.number).columns.tolist()
        categorical = [c for c in df.columns if df[c].dtype == 'object']
        
        if numeric and categorical:
            x = st.selectbox("Metric (e.g., Revenue, Profit)", numeric)
            y = st.selectbox("Dimension (e.g., Product, Region)", categorical)
            
            agg = df.groupby(y)[x].sum().reset_index().sort_values(x, ascending=False)
            fig1 = px.bar(agg, x=y, y=x, title=f"Total {x} by {y}")
            st.plotly_chart(fig1, use_container_width=True)
            
            top = agg.iloc[0][y]
            bottom = agg.iloc[-1][y]
            st.markdown(f"""
            💡 **Strategic Insight**:
            - **Opportunity**: Scale what works in **{top}** (highest {x})
            - **Risk**: Investigate underperformance in **{bottom}**
            - **Action**: Allocate resources to top performers; diagnose bottom
            """)
        
        st.subheader("2. Time-Series Trend (if date available)")
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            date_col = st.selectbox("Date column", date_cols)
            metric = st.selectbox("Metric for trend", numeric)
            if date_col and metric:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df_ts = df.dropna(subset=[date_col, metric])
                df_ts['Month'] = df_ts[date_col].dt.strftime('%Y-%m')
                trend = df_ts.groupby('Month')[metric].sum().reset_index()
                fig2 = px.line(trend, x='Month', y=metric, title=f"{metric} Trend Over Time")
                st.plotly_chart(fig2, use_container_width=True)
                
                if len(trend) > 2:
                    growth = (trend[metric].iloc[-1] - trend[metric].iloc[0]) / trend[metric].iloc[0]
                    st.markdown(f"""
                    📈 **Growth Strategy**:
                    - Overall change: **{growth:.1%}**
                    - If positive: Double down on current strategy
                    - If negative: Conduct root-cause analysis
                    """)
        
        st.subheader("3. Download Executive Summary")
        if st.button("Generate Business Insight Summary"):
            summary = {
                "dataset_rows": len(df),
                "key_metrics": numeric,
                "dimensions": categorical,
                "recommendation": "Focus on top-performing segments; monitor declining trends."
            }
            st.download_button(
                "📥 Download Summary (JSON)",
                json.dumps(summary, indent=2),
                "business_insights.json",
                "application/json"
            )

st.sidebar.markdown("---")
st.sidebar.markdown("🔒 **All data processed locally** — never stored")
st.sidebar.markdown("© 2025 Avash's Data Science Hub. All rights reserved.")
st.sidebar.markdown("🔗 Data Trainer available at: [avash-data-trainer.streamlit.app](https://data-trainer-2.streamlit.app)")
st.sidebar.markdown("🔗 Math Tool kit available at: [avash-math-toolkit.streamlit.app](https://avash-math-toolkit.streamlit.app)")
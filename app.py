import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Layout Setup
st.set_page_config(
    page_title="Fintech AI | Universal Bank Statement Categorizer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Modern Fintech Glassmorphism CSS Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #090d16 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-box {
        background: linear-gradient(90deg, rgba(79, 70, 229, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.4);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load AI Model and Vectorizer
@st.cache_resource
def load_ai_artifacts():
    try:
        model = joblib.load('transaction_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_ai_artifacts()

# App Header
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">Smart AI Bank Statement Categorizer</h1>
            <p class="header-subtitle">Universal Multi-Bank & PhonePe Statement Intelligence Engine</p>
        </div>
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 6px 14px; border-radius: 20px; color: #34d399; font-size: 0.85rem; font-weight: 600;">
            ● AI Engine Active
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Check model availability
if model is None or vectorizer is None:
    st.error("⚠️ Model files (`transaction_model.pkl` or `tfidf_vectorizer.pkl`) not found in the directory! Please place them alongside `app.py`.")
else:
    # Sidebar Upload Section
    st.sidebar.markdown("### 📥 Upload Statement")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, Excel, or Text Statement", 
        type=["csv", "xlsx", "xls", "txt"],
        help="Supports HDFC, SBI, ICICI, PhonePe, Google Pay, Paytm & Custom Statements."
    )

    if uploaded_file is not None:
        try:
            filename = uploaded_file.name
            # Read File based on format
            if filename.endswith('.csv') or filename.endswith('.txt'):
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep='\t')
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(uploaded_file)

            # --- SMART COLUMN AUTO-DETECTOR ---
            # Automatically detects description / narration / particulars column from any bank or app
            possible_desc_keywords = ['desc', 'narration', 'particular', 'detail', 'remark', 'note', 'transaction', 'merchant', 'paid to']
            desc_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in possible_desc_keywords):
                    desc_col = col
                    break
            
            # If no keyword matched, pick the column with the maximum average text length (likely description)
            if desc_col is None:
                text_lens = {col: df[col].astype(str).str.len().mean() for col in df.columns}
                desc_col = max(text_lens, key=text_lens.get)

            st.sidebar.success(f"File: **{filename}**")
            st.sidebar.info(f"Detected Text Column: **{desc_col}**")

            # --- RUN AI PREDICTION ---
            descriptions = df[desc_col].astype(str)
            X_vec = vectorizer.transform(descriptions)
            
            predictions = model.predict(X_vec)
            probabilities = model.predict_proba(X_vec)
            confidence_scores = np.max(probabilities, axis=1)
            
            df['Predicted_Category'] = predictions
            df['Confidence_Score'] = confidence_scores

            # Dashboard Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Executive Analytics", "⚠️ Low-Confidence Review", "📥 Export Clean Data"])

            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-label">Total Transactions</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{df["Predicted_Category"].nunique()}</div><div class="metric-label">Categories Found</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{df["Confidence_Score"].mean()*100:.1f}%</div><div class="metric-label">AI Confidence</div></div>', unsafe_allow_html=True)
                with c4:
                    low_count = len(df[df['Confidence_Score'] < 0.70])
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: #f43f5e;">{low_count}</div><div class="metric-label">Needs Review</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.write("### 📈 Category Spend / Transaction Distribution")
                cat_counts = df['Predicted_Category'].value_counts()
                st.bar_chart(cat_counts)

                st.write("### 📋 Categorized Transactions Data Preview")
                st.dataframe(df, use_container_width=True)

            with tab2:
                st.subheader("⚠️ Low-Confidence Audit & Review")
                st.write("Filter transactions where AI confidence is below your chosen threshold:")
                threshold = st.slider("Confidence Threshold Level", 0.0, 1.0, 0.70, 0.05)
                
                low_conf_df = df[df['Confidence_Score'] < threshold]
                st.write(f"Flagged Transactions: **{len(low_conf_df)}**")
                
                if len(low_conf_df) > 0:
                    st.dataframe(low_conf_df[[desc_col, 'Predicted_Category', 'Confidence_Score']], use_container_width=True)
                else:
                    st.success("🎉 Excellent! All transactions meet the confidence threshold.")

            with tab3:
                st.subheader("📥 Download Categorized Statement")
                st.write("Export your fully processed bank statement with AI categories and confidence scores.")
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Categorized CSV",
                    data=csv_data,
                    file_name=f"categorized_{filename.split('.')[0]}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing statement file: {e}")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 70px; background: rgba(30, 41, 59, 0.4); border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 16px;">
            <h3 style="color: #cbd5e1; margin-bottom: 10px;">📁 Ready for Statement Upload</h3>
            <p style="color: #64748b; font-size: 1rem;">Upload any bank statement or UPI app export (CSV, Excel, Text) from the sidebar to instantly activate AI categorization and predictions.</p>
        </div>
        """, unsafe_allow_html=True)
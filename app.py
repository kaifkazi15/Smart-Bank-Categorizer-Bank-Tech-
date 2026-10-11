import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="AuraPay | Global Treasury & AI Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# World-Class Neo-Banking CSS & Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 20%, #080e21 0%, #030611 100%);
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #030611;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(6, 11, 25, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .neo-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(16px);
    }
    .neo-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .neo-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .live-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .metric-container {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-lbl {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.7);
        border-radius: 10px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load AI Model safely
@st.cache_resource
def load_neo_models():
    try:
        model = joblib.load('transaction_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception:
        return None, None

model, vectorizer = load_neo_models()

# Hero Header
st.markdown("""
<div class="neo-header">
    <div>
        <h1 class="neo-title">AuraPay Intelligence</h1>
        <p class="neo-subtitle">Autonomous Multi-Bank & UPI Transaction Categorization Engine</p>
    </div>
    <div class="live-badge">
        <span style="width: 8px; height: 8px; background: #34d399; border-radius: 50%; display: inline-block;"></span>
        AI Core Online
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error("⚠️ Model artifacts (`transaction_model.pkl` or `tfidf_vectorizer.pkl`) missing in the root directory.")
else:
    st.sidebar.markdown("### 📥 Statement Ingestion")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Raw Bank or UPI Export",
        type=["csv", "xlsx", "xls", "txt"],
        help="Upload clean raw statements (Avoid uploading already categorized files)."
    )

    if uploaded_file is not None:
        try:
            filename = uploaded_file.name
            if filename.endswith('.csv') or filename.endswith('.txt'):
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_excel(uploaded_file)

            # --- SMART COLUMN DETECTOR (Ignores previous category artifacts) ---
            valid_cols = [c for c in df.columns if 'cat' not in str(c).lower()]
            search_space = valid_cols if valid_cols else df.columns

            keywords = ['desc', 'narration', 'particular', 'detail', 'remark', 'note', 'transaction', 'merchant', 'paid', 'item']
            desc_col = None
            
            for col in search_space:
                if any(kw in str(col).lower() for kw in keywords):
                    desc_col = col
                    break
            
            if desc_col is None:
                for col in search_space:
                    if df[col].dtype == 'object':
                        desc_col = col
                        break
            if desc_col is None:
                desc_col = search_space[0]

            st.sidebar.success(filename)
            st.sidebar.info(f"Active Field: `{desc_col}`")

            # Clean dataframe from any existing prediction columns to prevent conflicts
            for col in list(df.columns):
                if 'predicted' in col.lower() or col.lower() == 'category' or col.lower() == 'confidence':
                    df.drop(columns=[col], inplace=True)

            # Run Fresh AI Inference
            descriptions = df[desc_col].astype(str)
            X_vec = vectorizer.transform(descriptions)
            
            df['Category'] = model.predict(X_vec)
            df['Confidence'] = np.max(model.predict_proba(X_vec), axis=1)

            # Dashboard Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Executive Analytics", "🔍 Confidence & Audit", "📥 Verified Export"])

            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-container"><div class="metric-val">{len(df):,}</div><div class="metric-lbl">Transactions</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-container"><div class="metric-val" style="color: #38bdf8;">{df["Category"].nunique()}</div><div class="metric-lbl">Categories</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-container"><div class="metric-val" style="color: #34d399;">{df["Confidence"].mean()*100:.1f}%</div><div class="metric-lbl">Confidence</div></div>', unsafe_allow_html=True)
                with c4:
                    flagged = len(df[df['Confidence'] < 0.70])
                    st.markdown(f'<div class="metric-container"><div class="metric-val" style="color: #f43f5e;">{flagged}</div><div class="metric-lbl">Review Needed</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_left, col_right = st.columns([1.2, 1])
                custom_colors = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#34d399', '#fbbf24', '#f87171', '#a78bfa']

                with col_left:
                    st.subheader("Category Distribution (Donut Chart)")
                    cat_counts = df['Category'].value_counts().reset_index()
                    cat_counts.columns = ['Category', 'Count']

                    fig_donut = px.pie(cat_counts, names='Category', values='Count', hole=0.65, color_discrete_sequence=custom_colors)
                    fig_donut.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f1f5f9', family='Plus Jakarta Sans'),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                with col_right:
                    st.subheader("Category Breakdown")
                    fig_bar = px.bar(cat_counts, x='Category', y='Count', color='Category', color_discrete_sequence=custom_colors)
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f1f5f9', family='Plus Jakarta Sans'), showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Processed Ledger Table")
                st.dataframe(df, use_container_width=True)

            with tab2:
                st.subheader("Low Confidence Audit Queue")
                threshold = st.slider("Confidence Filter Level", 0.0, 1.0, 0.70, 0.05)
                audit_df = df[df['Confidence'] < threshold]
                st.markdown(f"**Flagged Entries:** {len(audit_df)}")
                if len(audit_df) > 0:
                    st.dataframe(audit_df, use_container_width=True)
                else:
                    st.success("All transactions clear confidence threshold.")

            with tab3:
                st.subheader("Export Verified Dataset")
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Audited CSV",
                    data=csv_bytes,
                    file_name=f"aurapay_clean_{filename.split('.')[0]}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error parsing file: {e}")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 90px 20px; background: rgba(15, 23, 42, 0.5); border: 2px dashed rgba(56, 189, 248, 0.2); border-radius: 20px;">
            <h3 style="color: #f1f5f9; margin-bottom: 10px; font-weight: 700;">Ready for Raw Statement Upload</h3>
            <p style="color: #64748b; font-size: 1rem;">Upload a clean raw statement (like the 50-day Excel or CSV file) to get precise AI categories for Movies, Shopping, Transport, and Food.</p>
        </div>
        """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="SEO Vector Similarity Tool", page_icon="🎯")

st.title("🎯 SEO Semantic Similarity Tool")
st.markdown("""
Upload your Screaming Frog 'Custom Extraction' CSV to identify keyword cannibalization using Vector Embeddings.
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. API Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.header("2. Similarity Thresholds")
    red_zone = st.slider("Red (Exact Duplicates)", 0.80, 1.00, 0.95)
    yellow_zone = st.slider("Yellow (Highly Similar)", 0.70, 0.95, 0.90)
    
    st.info("Note: $0.18 per 2,500 pages is based on 'text-embedding-3-small' model.")

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload your Screaming Frog CSV (must have 'Address' and 'Content' columns)", type="csv")

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)
    
    # Clean dataframe - adjust 'Content' column name if different
    if 'Content' in df.columns and 'Address' in df.columns:
        df = df[['Address', 'Content']].dropna()
        
        if st.button("Run Semantic Analysis"):
            client = OpenAI(api_key=api_key)
            
            with st.spinner('Generating Embeddings...'):
                try:
                    # Get embeddings from OpenAI
                    response = client.embeddings.create(
                        input=df['Content'].tolist(),
                        model="text-embedding-3-small"
                    )
                    embeddings = [record.embedding for record in response.data]
                    
                    # Calculate Similarity Matrix
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    # Find pairs
                    results = []
                    for i in range(len(df)):
                        for j in range(i + 1, len(df)):
                            score = similarity_matrix[i][j]
                            if score >= yellow_zone:
                                results.append({
                                    "URL A": df.iloc[i]['Address'],
                                    "URL B": df.iloc[j]['Address'],
                                    "Similarity Score": round(score, 4),
                                    "Status": "🔴 RED" if score >= red_zone else "🟡 YELLOW"
                                })
                    
                    results_df = pd.DataFrame(results)
                    
                    if not results_df.empty:
                        st.subheader("Analysis Results")
                        st.dataframe(results_df.sort_values(by="Similarity Score", ascending=False))
                        
                        # Download Button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Report", csv, "seo_analysis.csv", "text/csv")
                    else:
                        st.success("No significant cannibalization found! All pages look unique.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("CSV must contain 'Address' and 'Content' columns. Check your Screaming Frog export.")

elif not api_key and uploaded_file:
    st.warning("Please enter your OpenAI API Key in the sidebar.")

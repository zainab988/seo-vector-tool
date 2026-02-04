import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SEO Similarity Tool", page_icon="🎯")
st.title("🎯 SEO Semantic Similarity Tool")

# 1. Setup OpenAI Client
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to your Streamlit Secrets.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    ignore_params = st.checkbox("Ignore URLs with '?'", value=True)
    red_threshold = st.slider("Red Zone (Exact)", 0.90, 1.00, 0.95)
    yellow_threshold = st.slider("Yellow Zone (Similar)", 0.80, 0.95, 0.90)
    st.info("Filtering '?' URLs removes technical duplicates like sorting or tracking.")

uploaded_file = st.file_uploader("Upload your Screaming Frog CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 2. Smart Column Detection
    url_col = next((c for c in df.columns if 'Address' in c or 'URL' in c), None)
    content_col = next((c for c in df.columns if 'Extractor' in c or 'Content' in c), None)

    if url_col and content_col:
        # CLEANING: Force to string and handle missing values properly
        df[content_col] = df[content_col].fillna("").astype(str)
        
        # Remove truly empty rows
        df = df[df[content_col].str.strip() != ""]
        df = df[df[content_col].str.lower() != "nan"]
        
        # FILTER: Remove URLs with '?' if checked
        if ignore_params:
            initial_count = len(df)
            df = df[~df[url_col].str.contains(r'\?', na=False)]
            st.sidebar.write(f"Filtered {initial_count - len(df)} parameter URLs.")

        # TRUNCATE: Safety limit
        df[content_col] = df[content_col].apply(lambda x: x[:15000])
        
        st.success(f"Ready to analyze {len(df)} unique pages.")
        
        if st.button("Start Analysis"):
            if len(df) < 2:
                st.warning("Not enough pages left to compare after filtering!")
            else:
                with st.spinner('Generating Embeddings...'):
                    try:
                        # 3. Batch Processing
                        all_embeddings = []
                        content_list = df[content_col].tolist()
                        batch_size = 100 
                        
                        for i in range(0, len(content_list), batch_size):
                            batch = content_list[i:i + batch_size]
                            response = client.embeddings.create(
                                input=batch,
                                model="text-embedding-3-small"
                            )
                            all_embeddings.extend([r.embedding for r in response.data])
                        
                        # 4. Similarity Math
                        matrix = cosine_similarity(all_embeddings)
                        results = []
                        
                        for i in range(len(df)):
                            for j in range(i + 1, len(df)):
                                score = matrix[i][j]
                                if score >= yellow_threshold:
                                    results.append({
                                        "URL A": df.iloc[i][url_col],
                                        "URL B": df.iloc[j][url_col],
                                        "Similarity": round(score, 4),
                                        "Status": "🔴 RED" if score >= red_threshold else "🟡 YELLOW"
                                    })
                        
                        # 5. Output
                        if results:
                            res_df = pd.DataFrame(results).sort_values(by="Similarity", ascending=False)
                            st.subheader("Potential Cannibalization Found")
                            st.dataframe(res_df, use_container_width=True)
                            st.download_button("Download Report", res_df.to_csv(index=False), "seo_report.csv")
                        else:
                            st.success("No duplicates found!")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.error(f"Columns not found. I see: {list(df.columns)}")

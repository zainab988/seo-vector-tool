import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="SEO Similarity Tool", page_icon="🎯")
st.title("🎯 SEO Semantic Similarity Tool")

# 1. Setup OpenAI Client
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to your Streamlit Secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload your Screaming Frog CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 2. Smart Column Detection
    # Matches 'Address', 'Extractor 1 1', 'Content', etc.
    url_col = next((c for c in df.columns if 'Address' in c or 'URL' in c), None)
    content_col = next((c for c in df.columns if 'Extractor' in c or 'Content' in c), None)

    if url_col and content_col:
        # CLEANING: Remove rows where content is empty or not a string
        df[content_col] = df[content_col].astype(str)
        df = df[df[content_col].str.strip() != "nan"]
        df = df[df[content_col].str.strip() != ""]
        
        # TRUNCATE: Keep only the first 15,000 characters per page (safety limit)
        df[content_col] = df[content_col].apply(lambda x: x[:15000])
        
        st.success(f"Successfully loaded {len(df)} pages with content.")
        
        if st.button("Start Analysis"):
            with st.spinner('Generating Embeddings (Processing in batches)...'):
                try:
                    # 3. Batch Processing (Prevents BadRequestError)
                    all_embeddings = []
                    content_list = df[content_col].tolist()
                    batch_size = 100 # Process 100 rows at a time
                    
                    for i in range(0, len(content_list), batch_size):
                        batch = content_list[i:i + batch_size]
                        response = client.embeddings.create(
                            input=batch,
                            model="text-embedding-3-small"
                        )
                        all_embeddings.extend([r.embedding for r in response.data])
                    
                    # 4. Calculate Similarity
                    matrix = cosine_similarity(all_embeddings)
                    results = []
                    
                    for i in range(len(df)):
                        for j in range(i + 1, len(df)):
                            score = matrix[i][j]
                            if score >= 0.90:
                                results.append({
                                    "URL A": df.iloc[i][url_col],
                                    "URL B": df.iloc[j][url_col],
                                    "Similarity": round(score, 4),
                                    "Status": "🔴 RED" if score >= 0.95 else "🟡 YELLOW"
                                })
                    
                    # 5. Display Results
                    if results:
                        res_df = pd.DataFrame(results).sort_values(by="Similarity", ascending=False)
                        st.subheader("Analysis Results")
                        st.dataframe(res_df, use_container_width=True)
                        st.download_button("Download SEO Report", res_df.to_csv(index=False), "seo_duplicates.csv")
                    else:
                        st.balloons()
                        st.success("No significant cannibalization found! Your content is unique.")
                        
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.error(f"Required columns not found. Ensure your CSV has 'Address' and 'Extractor 1 1'. (Found: {list(df.columns)})")

import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎯 SEO Similarity Tool")

# This looks for your secret key automatically
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except:
    st.error("Missing API Key! Add 'OPENAI_API_KEY' to your Streamlit Secrets.")

uploaded_file = st.file_uploader("Upload Screaming Frog CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # This makes the tool smarter: it finds columns even if names are slightly different
    url_col = next((c for c in df.columns if 'Address' in c or 'URL' in c), None)
    content_col = next((c for c in df.columns if 'Content' in c or 'Extractor' in c), None)

    if url_col and content_col:
        df = df[[url_col, content_col]].dropna()
        df.columns = ['Address', 'Content']
        
        if st.button("Run Semantic Analysis"):
            with st.spinner('Analyzing...'):
                # 1. Get Embeddings
                response = client.embeddings.create(
                    input=df['Content'].tolist(),
                    model="text-embedding-3-small"
                )
                embeddings = [record.embedding for record in response.data]
                
                # 2. Compare Similarity
                matrix = cosine_similarity(embeddings)
                results = []
                for i in range(len(df)):
                    for j in range(i + 1, len(df)):
                        score = matrix[i][j]
                        if score >= 0.90:
                            results.append({
                                "URL A": df.iloc[i]['Address'],
                                "URL B": df.iloc[j]['Address'],
                                "Score": round(score, 4),
                                "Status": "🔴 RED" if score >= 0.95 else "🟡 YELLOW"
                            })
                
                if results:
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.success("No duplicates found!")
    else:
        st.error(f"Could not find Address or Content columns. Found: {list(df.columns)}")

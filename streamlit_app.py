import streamlit as st
import os
import xml.etree.ElementTree as ET
import re  # 正規表現モジュールをインポート
from sentence_transformers import SentenceTransformer, util

# XMLファイルから特定の要素のテキスト内容を取得する関数
def get_xml_text_content(xml_file, element_name='HASSEIJI_JOKYO_TXT'):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text_content = root.find(f'.//{element_name}').text
        return text_content
    except Exception as e:
        return ""

# SentenceTransformersを使用した類似度検索
def search_similar_documents(keyword, xml_directory='./xml'):
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    xml_texts = []
    xml_files = []

    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            xml_text = get_xml_text_content(file_path)
            if xml_text:
                xml_texts.append(xml_text)
                xml_files.append(filename)

    results = []
    if xml_texts:
        # キーワードとXML文書の埋め込みを計算
        keyword_embedding = model.encode([keyword], convert_to_tensor=True)
        doc_embeddings = model.encode(xml_texts, convert_to_tensor=True)
        # コサイン類似度の計算
        cosine_scores = util.pytorch_cos_sim(keyword_embedding, doc_embeddings)[0]

        similarity_scores = list(enumerate(cosine_scores.tolist()))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
        for idx, score in sorted_scores:
            results.append((xml_files[idx], score, xml_texts[idx]))

    return results

# 部分一致検索機能は変更なし
def search_documents_by_content(keyword, xml_directory='./xml'):
    results = []
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            xml_text = get_xml_text_content(file_path)
            if keyword.lower() in xml_text.lower():
                results.append((filename, xml_text))
    return results[:10]

# Streamlitアプリケーションのメイン関数
def main():
    st.title("XML Search and Similarity Tool")
    
    # キーワード入力
    keyword = st.text_area("Enter keyword for search or similarity search:", height=150)
    
    # 類似度検索ボタン
    if st.button("Search Similar Documents"):
        if keyword:
            results = search_similar_documents(keyword)
            for filename, score, text in results:
                # ファイル名から数字を抽出
                match = re.search(r'\d+', filename)
                if match:
                    trouble_id = match.group()
                    # 指定されたURLに数字を組み込む
                    url = f"http://www.nucia.jp/nucia/kn/KnTroubleView.do?troubleId={trouble_id}"
                    st.markdown(f"[{filename}: Score = {score:.4f}]({url})")
                    st.text_area("", text, height=300)
                else:
                    st.write(f"{filename}: Score = {score:.4f} (No ID found)")
                    st.text_area("", text, height=300)
        else:
            st.error("Please enter a keyword.")
    
    # 部分一致検索ボタン
    if st.button("Search Documents by Content"):
        if keyword:
            results = search_documents_by_content(keyword)
            for filename, text in results:
                # ファイル名から数字を抽出し、リンクを生成
                match = re.search(r'\d+', filename)
                if match:
                    trouble_id = match.group()
                    url = f"http://www.nucia.jp/nucia/kn/KnTroubleView.do?troubleId={trouble_id}"
                    st.markdown(f"[{filename}]({url})")
                    st.text_area("", text, height=300)
                else:
                    st.write(f"{filename} (No ID found)")
                    st.text_area("", text, height=300)
        else:
            st.error("Please enter a keyword.")

if __name__ == "__main__":
    main()

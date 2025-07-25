import streamlit as st, os, time, json, subprocess
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure the page
st.set_page_config(page_title="Expert-Tune")
st.title("🎓 Expert-Tune – ერთი ჩათი, ნულოვანი კოდი")

# Step 1 – choose domain and upload documents
domain = st.selectbox("თქვენი სფერო", ["იურისტი", "ფსიქოლოგი", "რესტორატორი", "სხვა"])
uploaded = st.file_uploader("📎 ატვირთეთ ნებისმიერი დოკუმენტი (PDF/Word/TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Step 2 – chat wizard state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "მოგესალმებით! რისი ექსპერტი ხართ?"}]

# Display existing chat messages
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# Handle chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Georgian-speaking AI question generator
    system = f"""
    You are a Georgian-speaking data-collector AI.
    Ask the expert (domain: {domain}) exactly 7 concise questions, one at a time, to extract facts and desired tone.
    Only ask the next question after receiving an answer.
    Finish with: «კითხვები დასრულდა».
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}] + st.session_state.messages,
        temperature=0.3
    )
    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

# Step 3 – build dataset and trigger finetuning
if st.button("🚀 დაიწყე ფაინ-ტუნინგი"):
    with st.spinner("კითხვა-პასუხი → JSONL → LoRA..."):
        # 1) Save Q/A pairs
        qa_pairs = st.session_state.messages[1::2]  # user answers
        data = []
        for i in range(0, len(qa_pairs), 2):
            if i + 1 < len(qa_pairs):
                prompt_text = f"შექმენი {domain} პასუხი საქართველოს კანონმდებლობის შესაბამისად.\n{qa_pairs[i]['content']}"
                completion_text = qa_pairs[i + 1]['content']
                data.append({"prompt": prompt_text, "completion": completion_text})
        # Write dataset to JSONL file
        with open("dataset.jsonl", "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # 2) Launch LoRA training
        subprocess.run(["python", "finetune.py"], check=True)
        st.success("მოდელი მზადაა! შეამოწმეთ ქვემოთ.")
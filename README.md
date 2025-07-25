# Expert-Finetune-One-Click

This repository contains a self‑contained Streamlit application that
guides a subject‑matter expert through a short question‑and‑answer
session, builds a training dataset from the responses and
automatically fine‑tunes a LoRA adapter on top of the
`unsloth/Meta‑Llama‑3.1‑8B‑Instruct` model. The resulting adapter can
then be used to provide domain‑specific responses in Georgian.

## File Structure

```
expert-tune/
├── app.py               # Streamlit front‑end and Q/A wizard
├── finetune.py          # LoRA training script (GPU)
├── Dockerfile           # For containerising the app
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Usage

1. **Environment Setup**: Install the required dependencies. The
   easiest way is to run inside the provided Docker container, but
   locally you can create a virtual environment and install
   dependencies from `requirements.txt`.

2. **API Key**: Create a `.env` file in the project root with your
   OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the App**: Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. **Answer Questions**: The app will ask you seven concise questions
   about your domain. Provide detailed answers in Georgian.

5. **Fine‑tune**: After answering, click the **🚀 დაიწყე ფაინ‑ტუნინგი**
   button. The Q/A pairs will be saved to `dataset.jsonl` and the
   LoRA fine‑tuning process will start automatically via
   `finetune.py`. When training completes, the adapter is saved to
   `lora_model`.

## Deployment

There are several ways to deploy this application:

* **Hugging Face Spaces** – Fork this repository on GitHub, set the
  `OPENAI_API_KEY` as a secret in the Space’s Variables, select a
  suitable GPU (e.g. A10G) and push. The app will run on
  `https://huggingface.co/spaces/<your-space>`.

* **Render** – Connect your GitHub repository to Render and create a
  new Web Service. Specify the build command (`pip install -r
  requirements.txt`) and the start command (`streamlit run app.py
  --server.port=$PORT --server.address=0.0.0.0`). Ensure you add
  `OPENAI_API_KEY` as an environment variable under **Environment**.

* **Modal** – If you have a `deploy.py` script configured, you can run
  the application via `modal run deploy.py`.

* **Google Colab** – To try it on Google Colab, download `app.py` and
  run it via `streamlit run app.py & npx localtunnel --port 8501` to
  expose it externally.

This repository and its contents are provided under their respective
licenses. Ensure you comply with the terms of the underlying models and
libraries when deploying a derivative model.
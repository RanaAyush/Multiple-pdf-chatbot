This is a Chat bot build using Retrieval Augmented Generation

Hi,

I have used Ollama to locally run create embeddings and store in FAISS vector store.
I used qwen2:1.5b model(You can use llama2 to increase speed and give better results) beacause that's because other models require higher RAMs.

Steps to run:

  1-> Download and Install Ollama from https://ollama.com/
  
  2-> Open terminal and type ollama run qwen2:1.5b (or lama2 or any other llm model you want to run).
  
  3-> Again type ollama run nomic-embed-text (to download embeddings)

  4-> To install dependencies Navigate this git repository:

   ---   (a) If want to create and run on your environment type(in terminal) --> python -m venv myenv 
      
   ---   (b) Then Activate it using -->  myenv/Scripts/activate

   ---   (c) Finally install dependencies --> pip install -r requirements.txt
  
  5->Finally run streamlit server using - streamlit run app.py

  6->Click on Process button to run directly for loaded pdfs

  7-> (optional) if you want to run it for your own pdfs just uncomment the the commented code and you can load your own pds.

Congratulations!!, it's working in your web browser

  Upload pdfs and click process and ask questions in prompt.

# ChatGPT-read-PDF
Small script to chat with your pdf with personal prompt template and static memory and vector

I used Chroma to vectorize and Spacy for an Italian dataset language

Remember that the chat history wants a tuple of conversations!

You just need to set a dir to save locally vector index, a dir to save locally chat history and set you OpenAI Key

Use 
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download it_core_news_lg

To install Ita repos, or visit the spacy site to download other language

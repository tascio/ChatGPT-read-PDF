import os
import spacy

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.memory import ConversationBufferMemory 
from langchain import PromptTemplate

KEY = " YOUR API KEY OPENAI"
os.environ['OPENAI_API_KEY'] = KEY

class Vector():
    def __init__(self):
        self.persist_directory = 'PERSISTENT VECTOR DIRECTORY'

        self.history = 'PERSISTENT CHAT HISTORY DIRECTORY'
        self.create_history()
        self.chat_history = {} #create the dict chat history to pass like conversational memory
        self.chat_history['conversation'] = []
        self.load_history_ff() #load data to dict chat history previous stored

        self.vectordb = ''
        self.embeddings = OpenAIEmbeddings()

    #If still doesn't exist, create it    
    def create_history(self):
        file = open(self.history, 'a')
        file.close()

    #update the conversation with AI storing it on file and update the chat_history dict
    def update_history(self, q, a):
        file = open(self.history, 'a')
        file.write(f"\n {q}@#,#@{a}")
        file.close()
        self.chat_history['conversation'].append({'HumanMessage':q, 'AIMessage':a})

    #load the history conversation from file    
    def load_history_ff(self):
        with open(self.history, 'r') as f:
            rows = f.readlines()
            for r in rows:
                try:
                    qa, an = r.split('@#,#@')
                    self.chat_history['conversation'].append({'HumanMessage':qa, 'AIMessage':an})
                except:
                    pass

    def load_db(self):
        self.vectordb = Chroma(embedding_function=self.embeddings, 
                               persist_directory=self.persist_directory)
        
    #load file, vectorize it, and store index pesistent    
    def load_pdf(self):
        pdf = input("Insert the full path 'yourfile.pdf'")
        pages = PyPDFLoader(pdf).load_and_split()
        spacy.load("it_core_news_lg")
        text_splitter = SpacyTextSplitter(chunk_size=1024, 
                                          chunk_overlap=200, 
                                          pipeline="it_core_news_lg")
        sections = text_splitter.split_documents(pages)
        self.vectordb = Chroma.from_documents(sections, 
                                              self.embeddings, 
                                              persist_directory=self.persist_directory)
        self.vectordb.persist()

    def getvectordb(self):
        return self.vectordb
    
    #return chat history in tuple how the prompt wants
    def getchathistory(self):
        tupla = []
        for r in self.chat_history['conversation']:
            tupla.append((f"{r['HumanMessage']}", f"{r['AIMessage']}"))
        return tupla
        
class QA():
    def __init__(self, vectordb):
        self.vectordb = vectordb
        self.PROMPT = self.define_prompt()
        self.qa = self.define_llm()
        
    def define_llm(self):
        chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo"),
                                                      retriever=self.vectordb.getvectordb().as_retriever(),
                                                      return_source_documents=True,
                                                      memory=ConversationBufferMemory(ai_prefix="Assistant",
                                                                                      human_prefix="Human",
                                                                                      memory_key='chat_history',
                                                                                      return_messages=True,
                                                                                      output_key='answer'),
                                                      combine_docs_chain_kwargs={"prompt":self.PROMPT},
                                                      verbose=True)
        return chain

    def define_prompt(self):
        template ="""Ricerca le informazioni per rispondere alle mie domande solo dai documenti forniti e dalle conversazioni precedenti avute con me in chat history, 
         se non trovi corrispondenze rispondi con 'no sacc'.
{context}

Human: {question}
Assistant:"""
        prompt = PromptTemplate(input_variables=['question','context'],
                                     template=template)
        return prompt
     
    def question(self, query):
        res = self.qa({"question": query, 'chat_history': self.vectordb.getchathistory()})
        print(res['answer'], '\n')
        self.vectordb.update_history(res['question'], res['answer']) #update the chat history dict

vectordb = Vector()
while (input('Vuoi inserire un documento? (rispondi con "y" si, "n" no)')) == 'y': #question to load new file pdf
    vectordb.load_pdf()
vectordb.load_db()

qa = QA(vectordb)

while True:
    question = input("Hai una domanda? ")
    qa.question(question)
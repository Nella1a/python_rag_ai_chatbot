import os
from dotenv import load_dotenv
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

DATA_PATH = "data_pdf/"
PDF_FILENAME = "TechCorp_Official_Employee_Handbook.pdf"
pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
print("Loading PDF document...")
loader = OpenDataLoaderPDFLoader(file_path=pdf_path, format="text")
document = loader.load()
print(document[0])


print("chunking text ....")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunk = text_splitter.split_documents(documents=document)
print(chunk[0])
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
loader = OpenDataLoaderPDFLoader(file_path=pdf_path, type="text")
document = loader.load()
print(document[0])
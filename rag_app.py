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


print("\n\n **** Loading PDF document... ***\n")
loader = OpenDataLoaderPDFLoader(file_path=pdf_path, format="text")
document = loader.load()
print(document[0])


print("\n\n *** chunking text .... ***\n")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents=document)
print(chunks[0])

print("\n\n *** Creating vector database ... ***\n")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Configure db to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # k = pull the two most relevant chunks for any given questiion

# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentence maximum and keep the answer concise.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)


user_question = "What days can I work from home?"
print(f"\nQuestion:", {user_question})

response = rag_chain.invoke(user_question)
print(f"Answer:", {response})
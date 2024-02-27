from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from config import settings
import PyPDF2
import os


os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key


def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def split_text_into_chunks(text, words_per_chunk):
    words = text.split()
    chunks = [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


def vectorstore_loader():
    pdf_file_path = "Michael_Igbomezie_Resume.pdf"  # Path to your PDF file
    words_per_chunk = 50  # Number of words per chunk

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_file_path)

    # Split text into chunks based on word count
    chunks = split_text_into_chunks(pdf_text, words_per_chunk)

    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_apikey)
    # vectordb = PineconeVectorStore.from_documents("pages", embeddings, index_name="my-resume")
    vectordb = PineconeVectorStore.from_texts(chunks, embeddings, index_name="my-resume")
    print("Successfully loaded pinecone vector store!")


def gpt_punctuator(information):
    '''Function is responsible for querying the GPT-3.5 model for analysis of a given content.'''

    client = OpenAI(
        # This is the default and can be omitted
        api_key=settings.openai_apikey,)

    #Prompt engineering message to be fed to the GPT model.
    messages = [
        {"role":"system","content":"you are a text analyst assistant"},
        {"role":"system","content":information}]

    #Creates the prompt to punctuate the subtitle extracted from the given video
    prompt_2 = "What is the Citizens' Voice Platform?"

    #Adds the prompts to the chat memory
    messages.append({"role": "user", "content": prompt_2},)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0.0,)
    
    #Response is extracted
    response = chat_completion.choices[0].message.content
    return (response)

    
def vectorstore_similaritysearch():
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_apikey)
    vectorstore = PineconeVectorStore(index_name="my-resume", embedding=embeddings)
    query = "What is the Citizens' Voice Platform?"
    docs = vectorstore.similarity_search(query, k=1)
    information = docs[0].page_content
    print(information)

    response = gpt_punctuator(information)
    print(response)


if __name__ == "__main__":
    # vectorstore_loader()
    vectorstore_similaritysearch()
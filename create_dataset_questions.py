import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

class GenerateDataSet:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()
        self.load_text()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_retrieval_chain()

    def load_text(self):
        print("Loading Text..")
        self.loader = PyMuPDFLoader('flexpod_ai_generative_ocp_m7.pdf')
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        print("Chunking Text..")
        self.text_splitter = SemanticChunker(self.embedding_model)
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        print("Storing in Chroma..")
        self.vectordb = Chroma.from_documents(self.docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def setup_conversation_retrieval_chain(self):
        print("Setup conversation..")
        llm = Ollama(model="phi3")
        self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 25}))

    def generate_questions(self, num_questions=250):
        questions = []
        prompt_text = "Assuming access to the Cisco Validated Design guide, generate a question, and only a question, about FlexPod, hardware components, storage, or networks, and, particularly, artificial intelligence. These should be questions a customer, partner, network architect, or data scientist might ask based on the information in the guide. ONLY GENERATE THE QUESTIONS. NO ANSWERS. Generate questions only and do not number the questions. Do not answer the questions."

        # Initialize chat history; empty if starting fresh each time
        chat_history = []

        for _ in range(num_questions):
            try:
                # Prepare input data with both question and chat_history
                input_data = {
                    'question': prompt_text,
                    'chat_history': chat_history  # Pass existing chat history
                }

                # Generate the question
                response_data = self.qa.invoke(input_data)

                # Extract and print the question
                new_question = response_data["answer"]
                if new_question not in questions:
                    print(new_question)  # Print the new question
                    questions.append(new_question)  # Store only if it's unique

            except Exception as e:
                print(f"Error generating question: {e}")
                continue  # Continue to the next iteration even if there's an error

        # Save the unique questions to a JSON file
        with open('generated_questions.json', 'w') as f:
            json.dump(questions, f, indent=4)
        print(f"Saved {len(questions)} unique questions to 'generated_questions.json'")

# Create an instance and use it
dataset_generator = GenerateDataSet()
dataset_generator.generate_questions()
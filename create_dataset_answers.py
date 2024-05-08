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

    def load_questions(self, filename):
        with open(filename, 'r') as file:
            self.questions = json.load(file)

    def save_to_jsonl(self, data, filename):
        with open(filename, 'w') as file:
            for entry in data:
                file.write(json.dumps(entry) + '\n')

    def generate_answer(self, question):
        # Generating a correct answer
        input_data = {
            'question': question,
            'chat_history': []  # Assuming starting fresh for each question
        }
        try:
            response_data = self.qa.invoke(input_data)
            return response_data["answer"]
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error in generating answer"

    def generate_rejected_answer(self, question):
        # Generating a rejected answer
        prompt_text = "Based on incorrect assumptions or errors, generate a plausible but incorrect answer for this question: " + question
        input_data = {
            'question': prompt_text,
            'chat_history': []  # Assuming starting fresh for each question
        }
        try:
            response_data = self.qa.invoke(input_data)
            return response_data["answer"]
        except Exception as e:
            print(f"Error generating rejected answer: {e}")
            return "Error in generating rejected answer"

    def generate_questions(self):
        self.load_questions('generated_questions.json')
        dataset = []
        for question in self.questions:
            chosen_answer = self.generate_answer(question)
            print(chosen_answer)
            rejected_answer = self.generate_rejected_answer(question)
            print(rejected_answer)
            dataset.append({
                'prompt': question,
                'chosen': chosen_answer,
                'rejected': rejected_answer
            })
        self.save_to_jsonl(dataset, 'training_dataset.jsonl')

# Create an instance and use it
dataset_generator = GenerateDataSet()
dataset_generator.generate_questions()
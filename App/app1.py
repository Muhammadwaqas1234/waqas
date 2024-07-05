import os
import logging
import asyncio  # Import asyncio here
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from gtts import gTTS
import base64
import re
import speech_recognition as sr
from llama_index.core import Settings
import openai
from llama_index.core.prompts.base import ChatPromptTemplate
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the 'static' directory to serve static files like logos
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

templates = Jinja2Templates(directory="templates")
openai.api_key = os.getenv("OPENAI_API_KEY")
messages = []
history = []

executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing

def load_data():
    try:
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, systemprompt="""Use the books in data file is source for the answer.Generate a valid 
                     and relevant answer to a query related to 
                     construction problems, ensure the answer is based strictly on the content of 
                     the book and not influenced by other sources. Do not hallucinate. The answer should 
                     be informative and fact-based.  """)
        service_content = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_content)
        return index
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True) if index else None

def query_chatbot(query_engine, user_question):
    try:
        response = query_engine.query(user_question)
        return response.response if response else None
    except Exception as e:
        logger.error(f"Error querying chatbot: {e}")
        return None

def initialize_chatbot(data_dir="./data", model="gpt-3.5-turbo", temperature=0.1):
    try:
        documents = SimpleDirectoryReader(data_dir).load_data()
        llm = OpenAI(model=model, temperature=temperature)

        additional_questions_prompt_str = (
            "Given the context below, generate only three additional question different from previous additional questions related to the user's query:\n"
            "Context:\n"
            "User Query: {query_str}\n"
            "Chatbot Response: \n"
        )

        new_context_prompt_str = (
            "We have the opportunity to only three generate additional question different from previous additional questions based on new context.\n"
            "New Context:\n"
            "User Query: {query_str}\n"
            "Chatbot Response: \n"
            "Given the new context, generate only three additional questions different at each time from previous additional questions related to the user's query."
            "If the context isn't useful, generate only three additional questions different at each from previous time from previous additional questions based on the original context.\n"
        )

        chat_text_qa_msgs = [
            (
                "system",
                """Generate only one additional question that facilitates deeper exploration of the main topic 
                discussed in the user's query and the chatbot's response. The questions should be relevant and
                  insightful, encouraging further discussion and exploration of the topic. Keep the questions concise 
                  and focused on different aspects of the main topic to provide a comprehensive understanding.""",
            ),
            ("user", additional_questions_prompt_str),
        ]
        text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

        chat_refine_msgs = [
            (
                "system",
                """Based on the user's question '{prompt}' and the chatbot's response '{response}', please 
                generate only one additional question related to the main topic. The questions should be 
                insightful and encourage further exploration of the main topic, providing a more comprehensive 
                understanding of the subject matter.""",
            ),
            ("user", new_context_prompt_str),
        ]
        refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            llm=llm,
        )

        return query_engine
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        return None

@app.get("/history", response_class=HTMLResponse)
def get_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "history": history})

def extract_document_section(response_text):
    pattern = r'\[SECTION_START\](.*?)\[SECTION_END\]'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_text.strip()

def generate_additional_questions(user_question, response_text, document_section, chat_engine):
    try:
        additional_questions = []
        for _ in range(1):
            additional_question = query_chatbot(initialize_chatbot(), user_question)
            additional_questions.append(additional_question if additional_question else None)
        return additional_questions
    except Exception as e:
        logger.error(f"Error generating additional questions: {e}")
        return []

async def generate_response(user_question, chat_engine):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, chat_engine.chat, user_question)
        if response:
            response_text = response.response
            document_section = extract_document_section(response_text)

            tts = gTTS(text=response_text, lang='en')
            tts.save('output.mp3')

            with open('output.mp3', 'rb') as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

            additional_question_text = generate_additional_questions(user_question, response_text, document_section, chat_engine)

            response_words = response_text.split()

            return response_words, document_section, audio_data, additional_question_text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
    return None, None, None, None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages, "history": history})

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request):
    try:
        data = await request.json()
        user_question = data["user_question"]
        response_words, document_section, audio_data, additional_question_text = await generate_response(user_question, chat_engine)
        
        if response_words:
            append_message('user', user_question)
            append_message('assistant', response_words, type='response')

            return {
                "response_words": response_words,
                "document_section": document_section,
                "audio_data": audio_data,
                "additional_question_text": additional_question_text
            }
        else:
            return JSONResponse(content={"error": "No response generated"}, status_code=500)

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred while processing your request."}, status_code=500)


@app.post("/additional_question", response_class=JSONResponse)
async def additional_question(request: Request):
    try:
        data = await request.json()
        additional_question_text = data["additional_question_text"]
        response_words, document_section, audio_data, additional_question_text = await generate_response(additional_question_text, chat_engine)
        append_message('user', additional_question_text)
        append_message('assistant', response_words, type='response')

        return {"response_words": response_words, "document_section": document_section, "audio_data": audio_data, "additional_question_text": additional_question_text}
    except Exception as e:
        logger.error(f"Error in /additional_question endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred while processing your request."}, status_code=500)

def append_message(role, message, type='message'):
    messages.append({"role": role, "content": message, "type": type})
    history.append({"role": role, "content": message, "type": type})

@app.post("/voice_command", response_class=JSONResponse)
async def voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        user_question = recognizer.recognize_google(audio)
        logger.info(f"User Command: {user_question}")
        response_words, document_section, audio_data, additional_question_text = await generate_response(user_question, chat_engine)
        append_message('user', user_question)
        append_message('assistant', response_words, type='response')

        return {"response_words": response_words, "document_section": document_section, "audio_data": audio_data, "additional_question_text": additional_question_text}
    except sr.UnknownValueError:
        logger.error("Could not understand audio")
        return {"error": "Could not understand audio"}
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return {"error": f"Could not request results from Google Speech Recognition service; {e}"}
    except Exception as e:
        logger.error(f"Error in /voice_command endpoint: {e}")
        return JSONResponse(content={"error": "An error occurred while processing your request."}, status_code=500)

def openai_generate_response(additional_question):    
    try:
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, systemprompt="""Use the books in data file is source for the answer.Generate a valid 
                     and relevant answer to a query related to 
                     construction problems, ensure the answer is based strictly on the content of 
                     the book and not influenced by other sources. Do not hallucinate. The answer should 
                     be informative and fact-based. """)
        answer = llm.generate_prompt(prompt=f"{additional_question}\n", max_tokens=100, temperature=0.8)
        return answer
    except Exception as e:
        logger.error(f"Error generating OpenAI response: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)

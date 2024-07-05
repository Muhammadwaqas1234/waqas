import os
import uuid
from datetime import datetime
import openai
import boto3
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from gtts import gTTS
from boto3.dynamodb.conditions import Key, Attr
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import base64
import stripe
from stripe.error import SignatureVerificationError
from llama_index.core.prompts.base import ChatPromptTemplate

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
static = Jinja2Templates(directory="static")
data = Jinja2Templates(directory="data")

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-2')
dynamodb_client = boto3.client('dynamodb', region_name='ap-southeast-2')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define Pydantic models
class UserRegisterForm(BaseModel):
    username: str
    email: str
    password: str

class UserLoginForm(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    user_question: str

class Feedback(BaseModel):
    feedback: str

# Create DynamoDB tables
def create_dynamodb_table(table_name, key_schema, attribute_definitions, provisioned_throughput, global_secondary_indexes=None):
    try:
        table_params = {
            'TableName': table_name,
            'KeySchema': key_schema,
            'AttributeDefinitions': attribute_definitions,
            'ProvisionedThroughput': provisioned_throughput
        }
        if global_secondary_indexes:
            table_params['GlobalSecondaryIndexes'] = global_secondary_indexes

        table = dynamodb.create_table(**table_params)
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Table {table_name} created successfully.")
    except dynamodb_client.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists.")

# Initialize required tables
create_dynamodb_table(
    'Users',
    key_schema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
    attribute_definitions=[{'AttributeName': 'id', 'AttributeType': 'S'}, {'AttributeName': 'email', 'AttributeType': 'S'}],
    provisioned_throughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10},
    global_secondary_indexes=[
        {
            'IndexName': 'email-index',
            'KeySchema': [{'AttributeName': 'email', 'KeyType': 'HASH'}],
            'Projection': {'ProjectionType': 'ALL'},
            'ProvisionedThroughput': {'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}
        }
    ]
)

create_dynamodb_table(
    'ChatHistory',
    key_schema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}, {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}],
    attribute_definitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}, {'AttributeName': 'timestamp', 'AttributeType': 'S'}],
    provisioned_throughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}
)

create_dynamodb_table(
    'Feedback',
    key_schema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}, {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}],
    attribute_definitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}, {'AttributeName': 'timestamp', 'AttributeType': 'S'}],
    provisioned_throughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}
)

# DynamoDB Table references
users_table = dynamodb.Table('Users')
chat_history_table = dynamodb.Table('ChatHistory')
feedback_table = dynamodb.Table('Feedback')

messages = []

# Utility functions
def appendMessage(role, message, type='message'):
    messages.append({"role": role, "content": message, "type": type})

pdf_dir = "./data"

def load_data():
    reader = SimpleDirectoryReader(pdf_dir, recursive=True)
    docs = reader.load_data()
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, systemprompt="""Use the books in data file as source for the answer. Generate a valid 
                 and relevant answer to a query related to 
                 construction problems, ensure the answer is based strictly on the content of 
                 the book and not influenced by other sources. Do not hallucinate. The answer should 
                 be informative and fact-based. """)
    service_content = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_content)
    return index

def query_chatbot(query_engine, user_question):
    response = query_engine.query(user_question)
    return response.response if response else None

def initialize_chatbot(pdf_dir = "./data", model="gpt-3.5-turbo", temperature=0.4):
    documents = SimpleDirectoryReader(pdf_dir).load_data()
    llm = OpenAI(model=model, temperature=temperature)

    additional_questions_prompt_str = (
        "Given the context below, generate only one additional question different from previous additional questions related to the user's query:\n"
        "Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
    )

    new_context_prompt_str = (
        "We have the opportunity to only one generate additional question different from previous additional questions based on new context.\n"
        "New Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
        "Given the new context, generate only one additional questions different at each time from previous additional questions related to the user's query."
        "If the context isn't useful, generate only one additional questions different at each from previous time from previous additional questions based on the original context.\n"
    )

    chat_text_qa_msgs = [
        (
            "system",
            """Generate only one additional question that facilitates deeper exploration of the main topic 
            discussed in the user's query and the chatbot's response. The question should be relevant and
              insightful, encouraging further discussion and exploration of the topic. Keep the question concise 
              and focused on different aspects of the main topic to provide a comprehensive understanding.""",
        ),
        ("user", additional_questions_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    chat_refine_msgs = [
        (
            "system",
            """Based on the user's question '{prompt}' and the chatbot's response '{response}', please 
            generate only one additional question related to the main topic. The question should be 
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

def generate_response(user_question):
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    response = chat_engine.chat(user_question)
    if response:
        response_text = response.response

        tts = gTTS(text=response_text, lang='en')
        tts.save('output.wav')

        with open('output.wav', 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

        additional_questions = generate_additional_questions(response_text)
        document_session = extract_document_section(response_text)

        return  response_text, additional_questions, audio_data, document_session

    return None, None, None, None

def generate_additional_questions(user_question):
    additional_questions = []
    words = ["1", "2", "3"]
    for word in words:
        question = query_chatbot(initialize_chatbot(), user_question)
        additional_questions.append(question if question else None)

    return additional_questions

def extract_text_from_pdf_page(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    text = page.get_text("text")
    return text

def extract_document_section(response_text, pdf_dir="./data"):
    pdf_texts = {}
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            page_texts = []
            for i in range(page_count):
                page_texts.append(extract_text_from_pdf_page(pdf_path, i))
            pdf_texts[filename] = page_texts

    response_paragraphs = response_text.split("\n\n")
    most_similar_page = None
    max_similarity = 0.0
    for pdf_name, page_texts in pdf_texts.items():
        for page_num, page_text in enumerate(page_texts):
            for response_paragraph in response_paragraphs:
                corpus = [page_text, response_paragraph]
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus)
                similarity = cosine_similarity(tfidf_matrix)[0][1]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_page = (pdf_name, page_num)

    if most_similar_page is not None:
        pdf_name, page_num = most_similar_page
        return pdf_texts[pdf_name][page_num]
    else:
        return "Question is out of documents"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    if 'username' in request.session:
        return templates.TemplateResponse("index.html", {"request": request, "messages": messages})
    return RedirectResponse(url="/login")

@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request, form_data: UserRegisterForm):
    user_type = "basic"
    response = users_table.query(
        IndexName='email-index',
        KeyConditionExpression=Key('email').eq(form_data.email)
    )
    if response['Items']:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered."})

    user_id = str(uuid.uuid4())
    registration_date = datetime.utcnow().isoformat()

    users_table.put_item(
        Item={
            "id": user_id,
            "first_name": form_data.first,
            "last_name": form_data.last,
            "username": form_data.username,
            "password": form_data.password,
            "email": form_data.email,
            "registration_date": registration_date,
            "user_type": user_type,
            "question_count": 0,
            "last_question_date": registration_date
        }
    )
    
    request.session["username"] = form_data.username
    request.session["user_id"] = user_id

    return RedirectResponse(url="/index", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, form_data: UserLoginForm):
    response = users_table.query(
        IndexName='email-index',
        KeyConditionExpression=Key('email').eq(form_data.email)
    )

    if response['Items']:
        request.session["username"] = response['Items'][0]['username']
        request.session["user_id"] = response['Items'][0]['id']
        return RedirectResponse(url="/index", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid email or password."})

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request, chat_request: ChatRequest):
    if 'username' not in request.session:
        raise HTTPException(status_code=403, detail="User not logged in")

    user_question = chat_request.user_question
    user_id = request.session['user_id']

    response = users_table.get_item(Key={'id': user_id})
    user = response.get('Item')
    if user:
        last_question_date = datetime.fromisoformat(user.get('last_question_date', '1970-01-01')).date()
        current_date = datetime.utcnow().date()

        if last_question_date < current_date:
            user['question_count'] = 0

        question_limit = 10 if user.get('user_type') == 'pro' else 5

        if user['question_count'] >= question_limit:
            return JSONResponse(status_code=403, content={"error": f"{user['user_type'].capitalize()} user has reached maximum question limit"})

        user['question_count'] += 1
        user['last_question_date'] = current_date.isoformat()
        users_table.put_item(Item=user)

        response_text, additional_questions, audio_data, document_session = generate_response(user_question)
        appendMessage('user', user_question)
        appendMessage('assistant', response_text, type='response')

        if additional_questions:
            for question in additional_questions:
                appendMessage("user", question)
                appendMessage('assistant', question, type='additional_question')

        chat_history_table.put_item(
            Item={
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_question": user_question,
                "chatbot_response": response_text
            }
        )

        return JSONResponse(content={"response_text": response_text, "additional_questions": additional_questions, "audio_data": audio_data, "document_session": document_session})

    raise HTTPException(status_code=404, detail="User not found")

@app.get("/change_password", response_class=HTMLResponse)
async def change_password_get(request: Request):
    if 'username' not in request.session:
        return RedirectResponse(url='/login')

    return templates.TemplateResponse("change_password.html", {"request": request})

@app.post("/change_password", response_class=HTMLResponse)
async def change_password_post(request: Request, current_password: str = Form(...), new_password: str = Form(...), confirm_password: str = Form(...)):
    if 'username' not in request.session:
        return RedirectResponse(url='/login')

    user_id = request.session['user_id']
    response = users_table.get_item(Key={'id': user_id})
    user = response.get('Item')

    if user:
        if user['password'] != current_password:
            return templates.TemplateResponse("change_password.html", {"request": request, "error": "Current password is incorrect"})

        if new_password != confirm_password:
            return templates.TemplateResponse("change_password.html", {"request": request, "error": "Passwords do not match"})

        user['password'] = new_password
        users_table.put_item(Item=user)

        return RedirectResponse(url='/account')

    return RedirectResponse(url='/index')

@app.get("/account", response_class=HTMLResponse)
async def account(request: Request):
    if 'username' not in request.session:
        return RedirectResponse(url='/login')

    user_id = request.session['user_id']
    response = users_table.get_item(Key={'id': user_id})
    user = response.get('Item')

    if user:
        user_data = {
            "username": user['username'],
            "email": user['email'],
            "password": user['password']
        }
        return templates.TemplateResponse("account.html", {"request": request, "user": user_data})
    return RedirectResponse(url='/index')

@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

def get_current_user(request: Request):
    username = request.session.get("username")
    user_id = request.session.get("user_id")
    if not username or not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not logged in")
    return {"username": username, "user_id": user_id}

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    response = chat_history_table.query(
        KeyConditionExpression=Key('user_id').eq(user_id),
        ScanIndexForward=False  # Descending order
    )
    items = response['Items']
    chat_history = [
        {
            "timestamp": item['timestamp'],
            "user_question": item['user_question'],
            "chatbot_response": item['chatbot_response']
        }
        for item in items
    ]
    return templates.TemplateResponse("history.html", {"request": request, "chat_history": chat_history})

@app.get("/support", response_class=HTMLResponse)
async def support_form(request: Request):
    return templates.TemplateResponse("support.html", {"request": request})

@app.post("/support", response_class=HTMLResponse)
async def support_submit(request: Request, message: str = Form(...), current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    feedback_table.put_item(Item={
        'user_id': user_id,
        'timestamp': str(datetime.utcnow()),
        'feedback': message
    })
    return templates.TemplateResponse("feedback_submitted.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('username', None)
    request.session.pop('user_id', None)
    return RedirectResponse(url='/login', status_code=status.HTTP_302_FOUND)

def handle_checkout_session(session):
    print("Handling checkout session...")
    customer_email = session['customer_details']['email']
    print(f"Customer email: {customer_email}")

    response = users_table.scan(FilterExpression=Attr('email').eq(customer_email))
    users = response['Items']

    if users:
        user = users[0]
        print(f"Found user: {user}")

        user['user_type'] = 'pro'
        users_table.put_item(Item=user)
        print(f"Updated user: {user}")
    else:
        print("User not found")

    print("Checkout session handling complete.")

@app.post("/webhook")
async def stripe_webhook(request: Request):
    print("Webhook received")
    payload = await request.body()
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        print("Event:", event)
    except ValueError as e:
        print("ValueError:", e)
        return JSONResponse(content={"success": False}, status_code=400)
    except SignatureVerificationError as e:
        print("SignatureVerificationError:", e)
        return JSONResponse(content={"success": False}, status_code=400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        handle_checkout_session(session)

    return JSONResponse(content={"success": True})

@app.get("/subscribe", response_class=HTMLResponse)
async def subscribe_form(request: Request):
    return templates.TemplateResponse("subscribe.html", {"request": request})

@app.post("/subscribe")
async def subscribe(request: Request, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    print(f"User ID from session: {user_id}")

    response = users_table.get_item(Key={'id': user_id})
    user = response.get('Item')
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    print(f"User found: {user}")

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=user['email'],
            line_items=[{
                'price': 'price_1PQOO3Gthr7AaSvU3fHuPOGN',
            }],
            mode='subscription',
            success_url=request.url_for('subscription_success'),
            cancel_url=request.url_for('subscription_cancel'),
        )

        print(f"Checkout session created: {checkout_session}")

        return JSONResponse(content={'checkout_session_id': checkout_session['id']})
    except Exception as e:
        print(f"Error creating checkout session: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=403)

@app.get("/subscription_success", response_class=HTMLResponse)
async def subscription_success(request: Request, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    response = users_table.get_item(Key={'id': user_id})
    user = response.get('Item')

    if user:
        user['user_type'] = 'pro'
        users_table.put_item(Item=user)

    return templates.TemplateResponse("subscription_success.html", {"request": request})

@app.get("/subscription_cancel", response_class=HTMLResponse)
async def subscription_cancel(request: Request):
    return templates.TemplateResponse("subscription_cancel.html", {"request": request})

@app.post("/feedback")
async def feedback(feedback: Feedback, current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']

    feedback_table.put_item(Item={
        'user_id': user_id,
        'timestamp': str(datetime.utcnow()),
        'feedback': feedback.feedback
    })

    return JSONResponse(content={"message": "Thank you for your feedback!"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

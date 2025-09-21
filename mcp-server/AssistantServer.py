# import os
# import logging
# import psycopg2
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from datetime import datetime, timedelta
# from apscheduler.schedulers.background import BackgroundScheduler
# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# from dotenv import load_dotenv
# import imaplib
# import nest_asyncio
# import re
# from rapidfuzz import fuzz
# import requests
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# import json
# from tenacity import retry, stop_after_attempt, wait_exponential
# from mcp.server.fastmcp import FastMCP

# # Apply nest_asyncio for async compatibility
# nest_asyncio.apply()

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# SHEET_ID = os.getenv("SHEET_ID")
# GOOGLE_CX = os.getenv("GOOGLE_CX")
# GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
# GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
# GMAIL_RECIPIENT = os.getenv("GMAIL_RECIPIENT")

# # Google services
# SERVICE_ACCOUNT_FILE = './app1-a5b21-2d18ba85db02.json'
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/tasks', 'https://www.googleapis.com/auth/calendar']
# try:
#     credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#     sheets_service = build('sheets', 'v4', credentials=credentials)
#     tasks_service = build('tasks', 'v1', credentials=credentials)
#     calendar_service = build('calendar', 'v3', credentials=credentials)
# except Exception as e:
#     logger.error(f"Google services error: {e}")
#     sheets_service = tasks_service = calendar_service = None

# # Database connection with retry
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def get_db_connection():
#     try:
#         conn = psycopg2.connect(
#             host=DB_HOST,
#             port=int(DB_PORT),
#             dbname=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             sslmode="require"
#         )
#         return conn
#     except psycopg2.Error as e:
#         logger.error(f"Database connection error: {e}")
#         raise

# def init_db():
#     conn = get_db_connection()
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS chat_history (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 query TEXT,
#                 context TEXT,
#                 response TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#             CREATE TABLE IF NOT EXISTS general_chat_history (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 query TEXT,
#                 response TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#             CREATE TABLE IF NOT EXISTS tasks (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 task_description TEXT,
#                 due_date TIMESTAMP,
#                 status TEXT DEFAULT 'pending',
#                 priority TEXT DEFAULT 'medium'
#             );
#             CREATE TABLE IF NOT EXISTS user_profiles (
#                 user_id TEXT PRIMARY KEY,
#                 email_filters TEXT DEFAULT 'ALL',
#                 reminder_preference TEXT DEFAULT 'tasks'
#             );
#         """)
#         cur.execute("INSERT INTO user_profiles (user_id) VALUES (%s) ON CONFLICT DO NOTHING", ("abhiram",))
#         conn.commit()
#     except psycopg2.Error as e:
#         logger.error(f"Database error: {e}")
#         raise
#     finally:
#         cur.close()
#         conn.close()

# init_db()

# # RAG setup with FAISS
# sample_docs = [
#     "Personal assistant manages emails via IMAP/SMTP, schedules tasks in Google Tasks, and updates Google Sheets daily.",
#     "Tasks are stored in PostgreSQL with fields: user_id, task_description, due_date, status, priority. Query via SQL for retrieval.",
#     "Google Search uses Custom Search API. Daily summaries are emailed at 6 AM with task and email updates.",
#     "Meetings can be scheduled using Google Calendar API with attendee emails and time slots.",
#     "LlamaIndex is a framework for building context-augmented generative AI applications, enabling RAG (Retrieval-Augmented Generation) by connecting LLMs to external data sources like documents, databases, or APIs."
# ]
# with open('./sample.txt', 'w') as f:
#     f.write("\n".join(sample_docs))
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# docs = TextLoader("./sample.txt").load()
# vectorstore = FAISS.from_documents(docs, embeddings)
# retriever = vectorstore.as_retriever()

# # LLM setup
# try:
#     rag_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
#     response = rag_llm.invoke("Test connection")
#     logger.info(f"Gemini connected: {response.content}")
# except Exception as e:
#     logger.error(f"Gemini failed: {e}")
#     raise RuntimeError("Cannot connect to Gemini. Check API key and quota at https://aistudio.google.com/app/apikey.")

# # MCP server setup
# mcp = FastMCP("Assistant")

# @mcp.tool()
# async def check_important_emails(user_id: str) -> str:
#     """Fetch recent email subjects from Gmail inbox for a given user."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT email_filters FROM user_profiles WHERE user_id = %s", (user_id,))
#         row = cur.fetchone()
#         filters = row[0] if row else 'ALL'
#         cur.close()
#         conn.close()
#         mail = imaplib.IMAP4_SSL("imap.gmail.com")
#         mail.login(GMAIL_USERNAME, GMAIL_PASSWORD)
#         mail.select("inbox")
#         status, messages = mail.search(None, filters)
#         if status != 'OK':
#             logger.warning(f"IMAP search failed: {messages}. Falling back to ALL.")
#             status, messages = mail.search(None, 'ALL')
#             if status != 'OK':
#                 raise Exception(f"IMAP fallback search failed: {messages}")
#         email_ids = messages[0].split()
#         summaries = []
#         for email_id in email_ids[-5:]:
#             status, msg_data = mail.fetch(email_id, "(RFC822)")
#             if status != 'OK':
#                 continue
#             msg = msg_data[0][1].decode("utf-8", errors="ignore")
#             subject = next((line.split(": ", 1)[1] for line in msg.split("\n") if line.startswith("Subject:")), "No Subject")
#             summaries.append(f"Subject: {subject}")
#         mail.logout()
#         return "\n".join(summaries) or "No emails found."
#     except Exception as e:
#         logger.error(f"Error checking emails: {e}")
#         return f"Failed to check emails: {str(e)}. Ensure IMAP is enabled: https://mail.google.com/mail/u/0/#settings/fwdandpop"

# @mcp.tool()
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def send_email(subject: str, body: str, recipient: str = None) -> str:
#     """Send an email via Gmail SMTP."""
#     try:
#         msg = MIMEMultipart()
#         msg['From'] = GMAIL_USERNAME
#         msg['To'] = recipient or GMAIL_RECIPIENT
#         msg['Subject'] = subject
#         msg.attach(MIMEText(body, 'plain'))
#         with smtplib.SMTP('smtp.gmail.com', 587) as server:
#             server.starttls()
#             server.login(GMAIL_USERNAME, GMAIL_PASSWORD)
#             server.sendmail(msg['From'], msg['To'], msg.as_string())
#         return f"Email sent to {msg['To']}."
#     except Exception as e:
#         logger.error(f"Error sending email: {e}")
#         return f"Failed to send email: {str(e)}"

# @mcp.tool()
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def perform_google_search(query: str) -> str:
#     """Perform a Google search using Custom Search API."""
#     try:
#         url = "https://www.googleapis.com/customsearch/v1"
#         params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         results = response.json().get("items", [])
#         formatted_results = "\n".join([f"{i+1}. {result['title']} - {result['link']}" for i, result in enumerate(results[:5])])
#         return formatted_results or "No search results found."
#     except Exception as e:
#         logger.error(f"Search error: {e}")
#         return f"Failed to perform search: {str(e)}. Verify API key and CX at https://console.developers.google.com"

# @mcp.tool()
# async def query_database(query: str, user_id: str) -> str:
#     """Execute SQL queries with typo correction."""
#     try:
#         corrections = {
#             "selct": "select", "lall": "all", "fromm": "from", "spreed": "spread",
#             "sheets": "sheet", "tabel": "table", "chatt_history": "chat_history", "delect": "drop"
#         }
#         corrected_query = query.lower()
#         for wrong, right in corrections.items():
#             if wrong in corrected_query:
#                 corrected_query = corrected_query.replace(wrong, right)
#         if re.search(r"\btop\s+\d+", corrected_query):
#             limit = re.search(r"\d+", corrected_query).group()
#             corrected_query = corrected_query.replace("top", "select").replace("records", "") + f" LIMIT {limit}"
#         if "drop table" in corrected_query:
#             return "WARNING: Dropping table will delete all data. Reply 'confirm DROP TABLE' to proceed."
#         conn = get_db_connection()
#         cur = conn.cursor()
#         if corrected_query.lower().startswith("select * from tables") or "tables in my database" in corrected_query.lower():
#             cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
#             tables = cur.fetchall()
#             cur.close()
#             conn.close()
#             return f"Tables in database:\n" + "\n".join([row[0] for row in tables]) or "No tables found."
#         cur.execute(corrected_query)
#         if corrected_query.lower().startswith("select"):
#             results = cur.fetchall()
#             columns = [desc[0] for desc in cur.description]
#             formatted_results = "\n".join([str(dict(zip(columns, row))) for row in results])
#             cur.close()
#             conn.close()
#             return f"Database query results:\n{formatted_results}" if results else "No results found."
#         else:
#             conn.commit()
#             cur.close()
#             conn.close()
#             return "Query executed successfully."
#     except psycopg2.Error as e:
#         logger.error(f"Database query error: {e}")
#         return f"Failed to execute query: {str(e)}"

# @mcp.tool()
# async def get_task_insights(user_id: str) -> str:
#     """Get task statistics by status."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT status, COUNT(*), 
#                    ROUND(AVG(EXTRACT(EPOCH FROM (due_date - CURRENT_TIMESTAMP))/86400)::numeric, 1) 
#             FROM tasks WHERE user_id = %s GROUP BY status
#         """, (user_id,))
#         stats = cur.fetchall()
#         insights = f"Task Insights for {user_id}:\n" + "\n".join([f"{row[0].title()}: {row[1]} tasks, Avg days to due: {row[2]}" for row in stats])
#         cur.close()
#         conn.close()
#         return insights or "No task insights available."
#     except Exception as e:
#         logger.error(f"Error fetching insights: {e}")
#         return f"Failed to fetch insights: {str(e)}"

# @mcp.tool()
# async def add_reminder(user_id: str, task_description: str, due_date: str = None) -> str:
#     """Add a task to database,   Tasks, and Sheets."""
#     try:
#         due = datetime.strptime(due_date, "%Y-%m-%d") if due_date else datetime.now() + timedelta(days=1)
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT reminder_preference FROM user_profiles WHERE user_id = %s", (user_id,))
#         row = cur.fetchone()
#         preference = row[0] if row else "tasks"
#         cur.execute("INSERT INTO tasks (user_id, task_description, due_date) VALUES (%s, %s, %s)", (user_id, task_description, due))
#         conn.commit()
#         cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#         tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
#         sheets_result = await update_sheets(user_id, json.dumps(tasks))
#         cur.close()
#         conn.close()
#         if preference == "tasks" and tasks_service:
#             task = {'title': task_description, 'due': due.isoformat() + 'Z'}
#             tasklist = tasks_service.tasklists().list().execute().get('items', [])
#             tasklist_id = tasklist[0]['id'] if tasklist else None
#             if tasklist_id:
#                 tasks_service.tasks().insert(tasklist=tasklist_id, body=task).execute()
#                 return f"Reminder added to Google Tasks.\n{sheets_result}"
#         return await send_email("Task Reminder", f"Reminder: {task_description} due on {due}", GMAIL_RECIPIENT) + f"\n{sheets_result}"
#     except Exception as e:
#         logger.error(f"Error adding reminder: {e}")
#         return f"Failed to add reminder: {str(e)}"

# @mcp.tool()
# async def delete_task(user_id: str, task_description: str) -> str:
#     """Delete a task from the database."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT id, task_description FROM tasks WHERE user_id = %s AND task_description ILIKE %s", (user_id, f"%{task_description}%"))
#         task = cur.fetchone()
#         if not task:
#             cur.close()
#             conn.close()
#             return f"No task found matching: {task_description}"
#         task_id = task[0]
#         cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
#         conn.commit()
#         cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#         tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
#         sheets_result = await update_sheets(user_id, json.dumps(tasks))
#         cur.close()
#         conn.close()
#         return f"Task '{task_description}' deleted.\n{sheets_result}"
#     except Exception as e:
#         logger.error(f"Error deleting task: {e}")
#         return f"Failed to delete task: {str(e)}"

# @mcp.tool()
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def update_sheets(user_id: str, tasks_json: str) -> str:
#     """Update Google Sheets with tasks."""
#     if not sheets_service:
#         return "Sheets service unavailable."
#     try:
#         tasks_list = json.loads(tasks_json)
#         spreadsheet_id = SHEET_ID
#         body = {'values': [[t['description'], t['due_date'], t['status'], t['priority']] for t in tasks_list]}
#         sheets_service.spreadsheets().values().update(
#             spreadsheetId=spreadsheet_id,
#             range="Sheet1!A1:D",
#             valueInputOption="RAW",
#             body=body
#         ).execute()
#         return "Spreadsheet updated successfully."
#     except Exception as e:
#         logger.error(f"Error updating Sheets: {e}")
#         return f"Failed to update Sheets: {str(e)}"

# @mcp.tool()
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def retrieve_sheets_data(user_id: str) -> str:
#     """Retrieve tasks from Google Sheets."""
#     if not sheets_service:
#         return "Sheets service unavailable."
#     try:
#         spreadsheet_id = SHEET_ID
#         result = sheets_service.spreadsheets().values().get(
#             spreadsheetId=spreadsheet_id,
#             range="Sheet1!A1:D"
#         ).execute()
#         values = result.get('values', [])
#         if not values:
#             return "No tasks found in the spreadsheet."
#         formatted_tasks = "\n".join([f"Task: {row[0]}, Due: {row[1]}, Status: {row[2]}, Priority: {row[3]}" for row in values if len(row) >= 4])
#         return f"Tasks in spreadsheet:\n{formatted_tasks}"
#     except Exception as e:
#         logger.error(f"Error retrieving Sheets data: {e}")
#         return f"Failed to retrieve Sheets data: {str(e)}"

# @mcp.tool()
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def schedule_meeting(attendee_email: str, time_str: str, description: str = "Meeting") -> str:
#     """Schedule a meeting using Google Calendar."""
#     if not calendar_service:
#         return "Calendar service unavailable."
#     try:
#         time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))|(\d{1,2}:\d{2}\s*hrs)", time_str, re.IGNORECASE)
#         if not time_match:
#             return "Invalid time format. Use '6:00 PM' or '18:00 hrs'."
#         time_str = time_match.group(0).replace("hrs", "").strip()
#         time = datetime.strptime(time_str, "%I:%M %p") if "AM" in time_str.upper() or "PM" in time_str.upper() else datetime.strptime(time_str, "%H:%M")
#         time = time.replace(year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
#         end_time = time + timedelta(hours=1)
#         event = {
#             'summary': description,
#             'start': {'dateTime': time.isoformat(), 'timeZone': 'Asia/Kolkata'},
#             'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
#             'attendees': [{'email': attendee_email}],
#         }
#         event = calendar_service.events().insert(calendarId='primary', body=event).execute()
#         return f"Meeting scheduled with {attendee_email} at {time_str}."
#     except Exception as e:
#         logger.error(f"Error scheduling meeting: {e}")
#         return f"Failed to schedule meeting: {str(e)}"

# @mcp.tool()
# async def summarize_tasks(user_id: str) -> str:
#     """Summarize pending tasks."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#         tasks = cur.fetchall()
#         cur.close()
#         conn.close()
#         if not tasks:
#             return "No pending tasks found."
#         formatted_tasks = "\n".join([f"Task: {row[0]}, Due: {row[1].strftime('%Y-%m-%d')}, Status: {row[2]}, Priority: {row[3]}" for row in tasks])
#         return f"Pending tasks:\n{formatted_tasks}"
#     except Exception as e:
#         logger.error(f"Error summarizing tasks: {e}")
#         return f"Failed to summarize tasks: {str(e)}"

# @mcp.tool()
# async def rag_query(query: str) -> str:
#     """Handle RAG queries with FAISS and LLM."""
#     try:
#         nodes = retriever.invoke(query)
#         context = "\n".join([n.page_content for n in nodes])
#         prompt = f"Using only the following context, answer '{query}':\n{context}\nIf the context doesn't fully answer, say so."
#         response = rag_llm.invoke(prompt).content
#         return response
#     except Exception as e:
#         logger.error(f"RAG error: {e}")
#         return f"RAG failed: {str(e)}"

# def send_daily_summary():
#     """Send daily summary email with tasks and emails (background task, not a tool)."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT user_id FROM user_profiles")
#         users = cur.fetchall()
#         for user_tuple in users:
#             user_id = user_tuple[0]
#             email_summary = asyncio.run(check_important_emails(user_id))
#             cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#             tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
#             task_summary = "\n".join([f"{t['description']} (Due: {t['due_date']}, Priority: {t['priority']})" for t in tasks]) or "No pending tasks."
#             summary = f"Daily Summary (6 AM):\n\nImportant Emails:\n{email_summary}\n\nTasks:\n{task_summary}"
#             asyncio.run(send_email("Daily Task Summary", summary))
#             asyncio.run(update_sheets(user_id, json.dumps(tasks)))
#         cur.close()
#         conn.close()
#     except Exception as e:
#         logger.error(f"Daily summary error: {e}")

# # Start background scheduler for daily summary
# scheduler = BackgroundScheduler()
# scheduler.add_job(send_daily_summary, 'date', run_date=datetime.now() + timedelta(seconds=30))
# scheduler.start()

# if __name__ == "__main__":
#     mcp.run(transport="stdio")


import os
import logging
# import psycopg2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from googleapiclient.discovery import build
from google.oauth2 import service_account
from dotenv import load_dotenv
import imaplib
import nest_asyncio
import re
from rapidfuzz import fuzz
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from mcp.server.fastmcp import FastMCP

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CX = os.getenv("GOOGLE_CX")
GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
GMAIL_RECIPIENT = os.getenv("GMAIL_RECIPIENT")

# Google services
SERVICE_ACCOUNT_FILE = './app1-a5b21-2d18ba85db02.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/tasks', 'https://www.googleapis.com/auth/calendar']
try:
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    sheets_service = build('sheets', 'v4', credentials=credentials)
    tasks_service = build('tasks', 'v1', credentials=credentials)
    calendar_service = build('calendar', 'v3', credentials=credentials)
except Exception as e:
    logger.error(f"Google services error: {e}")
    sheets_service = tasks_service = calendar_service = None

# # Database connection with retry
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def get_db_connection():
#     try:
#         conn = psycopg2.connect(
#             host=DB_HOST,
#             port=int(DB_PORT),
#             dbname=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             sslmode="require"
#         )
#         return conn
#     except psycopg2.Error as e:
#         logger.error(f"Database connection error: {e}")
#         raise
#
# def init_db():
#     conn = get_db_connection()
#     try:
#         cur = conn.cursor()
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS chat_history (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 query TEXT,
#                 context TEXT,
#                 response TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#             CREATE TABLE IF NOT EXISTS general_chat_history (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 query TEXT,
#                 response TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#             CREATE TABLE IF NOT EXISTS tasks (
#                 id SERIAL PRIMARY KEY,
#                 user_id TEXT,
#                 task_description TEXT,
#                 due_date TIMESTAMP,
#                 status TEXT DEFAULT 'pending',
#                 priority TEXT DEFAULT 'medium'
#             );
#             CREATE TABLE IF NOT EXISTS user_profiles (
#                 user_id TEXT PRIMARY KEY,
#                 email_filters TEXT DEFAULT 'ALL',
#                 reminder_preference TEXT DEFAULT 'tasks'
#             );
#         """)
#         cur.execute("INSERT INTO user_profiles (user_id) VALUES (%s) ON CONFLICT DO NOTHING", ("abhiram",))
#         conn.commit()
#     except psycopg2.Error as e:
#         logger.error(f"Database error: {e}")
#         raise
#     finally:
#         cur.close()
#         conn.close()
#
# init_db()

# RAG setup with FAISS
sample_docs = [
    "Personal assistant manages emails via IMAP/SMTP, schedules tasks in Google Tasks, and updates Google Sheets daily.",
    # "Tasks are stored in PostgreSQL with fields: user_id, task_description, due_date, status, priority. Query via SQL for retrieval.",
    "Google Search uses Custom Search API. Daily summaries are emailed at 6 AM with task and email updates.",
    "Meetings can be scheduled using Google Calendar API with attendee emails and time slots.",
    "LlamaIndex is a framework for building context-augmented generative AI applications, enabling RAG (Retrieval-Augmented Generation) by connecting LLMs to external data sources like documents, databases, or APIs."
]
with open('./sample.txt', 'w') as f:
    f.write("\n".join(sample_docs))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = TextLoader("./sample.txt").load()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# LLM setup
try:
    rag_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    response = rag_llm.invoke("Test connection")
    logger.info(f"Gemini connected: {response.content}")
except Exception as e:
    logger.error(f"Gemini failed: {e}")
    raise RuntimeError("Cannot connect to Gemini. Check API key and quota at https://aistudio.google.com/app/apikey.")

# MCP server setup
mcp = FastMCP("Assistant")

@mcp.tool()
async def check_important_emails(user_id: str) -> str:
    """Fetch recent email subjects from Gmail inbox for a given user."""
    try:
        # conn = get_db_connection()
        # cur = conn.cursor()
        # cur.execute("SELECT email_filters FROM user_profiles WHERE user_id = %s", (user_id,))
        # row = cur.fetchone()
        # filters = row[0] if row else 'ALL'
        # cur.close()
        # conn.close()
        filters = 'ALL'  # Default filter since no database
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USERNAME, GMAIL_PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, filters)
        if status != 'OK':
            logger.warning(f"IMAP search failed: {messages}. Falling back to ALL.")
            status, messages = mail.search(None, 'ALL')
            if status != 'OK':
                raise Exception(f"IMAP fallback search failed: {messages}")
        email_ids = messages[0].split()
        summaries = []
        for email_id in email_ids[-5:]:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != 'OK':
                continue
            msg = msg_data[0][1].decode("utf-8", errors="ignore")
            subject = next((line.split(": ", 1)[1] for line in msg.split("\n") if line.startswith("Subject:")), "No Subject")
            summaries.append(f"Subject: {subject}")
        mail.logout()
        return "\n".join(summaries) or "No emails found."
    except Exception as e:
        logger.error(f"Error checking emails: {e}")
        return f"Failed to check emails: {str(e)}. Ensure IMAP is enabled: https://mail.google.com/mail/u/0/#settings/fwdandpop"

@mcp.tool()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def send_email(subject: str, body: str, recipient: str = None) -> str:
    """Send an email via Gmail SMTP."""
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USERNAME
        msg['To'] = recipient or GMAIL_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_USERNAME, GMAIL_PASSWORD)
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        return f"Email sent to {msg['To']}."
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return f"Failed to send email: {str(e)}"

@mcp.tool()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def perform_google_search(query: str) -> str:
    """Perform a Google search using Custom Search API."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query}
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        formatted_results = "\n".join([f"{i+1}. {result['title']} - {result['link']}" for i, result in enumerate(results[:5])])
        return formatted_results or "No search results found."
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Failed to perform search: {str(e)}. Verify API key and CX at https://console.developers.google.com"

# @mcp.tool()
# async def query_database(query: str, user_id: str) -> str:
#     """Execute SQL queries with typo correction."""
#     try:
#         corrections = {
#             "selct": "select", "lall": "all", "fromm": "from", "spreed": "spread",
#             "sheets": "sheet", "tabel": "table", "chatt_history": "chat_history", "delect": "drop"
#         }
#         corrected_query = query.lower()
#         for wrong, right in corrections.items():
#             if wrong in corrected_query:
#                 corrected_query = corrected_query.replace(wrong, right)
#         if re.search(r"\btop\s+\d+", corrected_query):
#             limit = re.search(r"\d+", corrected_query).group()
#             corrected_query = corrected_query.replace("top", "select").replace("records", "") + f" LIMIT {limit}"
#         if "drop table" in corrected_query:
#             return "WARNING: Dropping table will delete all data. Reply 'confirm DROP TABLE' to proceed."
#         conn = get_db_connection()
#         cur = conn.cursor()
#         if corrected_query.lower().startswith("select * from tables") or "tables in my database" in corrected_query.lower():
#             cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
#             tables = cur.fetchall()
#             cur.close()
#             conn.close()
#             return f"Tables in database:\n" + "\n".join([row[0] for row in tables]) or "No tables found."
#         cur.execute(corrected_query)
#         if corrected_query.lower().startswith("select"):
#             results = cur.fetchall()
#             columns = [desc[0] for desc in cur.description]
#             formatted_results = "\n".join([str(dict(zip(columns, row))) for row in results])
#             cur.close()
#             conn.close()
#             return f"Database query results:\n{formatted_results}" if results else "No results found."
#         else:
#             conn.commit()
#             cur.close()
#             conn.close()
#             return "Query executed successfully."
#     except psycopg2.Error as e:
#         logger.error(f"Database query error: {e}")
#         return f"Failed to execute query: {str(e)}"
#
# @mcp.tool()
# async def get_task_insights(user_id: str) -> str:
#     """Get task statistics by status."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT status, COUNT(*), 
#                    ROUND(AVG(EXTRACT(EPOCH FROM (due_date - CURRENT_TIMESTAMP))/86400)::numeric, 1) 
#             FROM tasks WHERE user_id = %s GROUP BY status
#         """, (user_id,))
#         stats = cur.fetchall()
#         insights = f"Task Insights for {user_id}:\n" + "\n".join([f"{row[0].title()}: {row[1]} tasks, Avg days to due: {row[2]}" for row in stats])
#         cur.close()
#         conn.close()
#         return insights or "No task insights available."
#     except Exception as e:
#         logger.error(f"Error fetching insights: {e}")
#         return f"Failed to fetch insights: {str(e)}"

@mcp.tool()
async def add_reminder(user_id: str, task_description: str, due_date: str = None) -> str:
    """Add a task to Google Tasks and Sheets (database commented out)."""
    try:
        due = datetime.strptime(due_date, "%Y-%m-%d") if due_date else datetime.now() + timedelta(days=1)
        # conn = get_db_connection()
        # cur = conn.cursor()
        # cur.execute("SELECT reminder_preference FROM user_profiles WHERE user_id = %s", (user_id,))
        # row = cur.fetchone()
        # preference = row[0] if row else "tasks"
        # cur.execute("INSERT INTO tasks (user_id, task_description, due_date) VALUES (%s, %s, %s)", (user_id, task_description, due))
        # conn.commit()
        # cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
        # tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
        preference = "tasks"  # Default since no database
        tasks = [{'description': task_description, 'due_date': due.strftime('%Y-%m-%d'), 'status': 'pending', 'priority': 'medium'}]
        sheets_result = await update_sheets(user_id, json.dumps(tasks))
        # cur.close()
        # conn.close()
        if preference == "tasks" and tasks_service:
            task = {'title': task_description, 'due': due.isoformat() + 'Z'}
            tasklist = tasks_service.tasklists().list().execute().get('items', [])
            tasklist_id = tasklist[0]['id'] if tasklist else None
            if tasklist_id:
                tasks_service.tasks().insert(tasklist=tasklist_id, body=task).execute()
                return f"Reminder added to Google Tasks.\n{sheets_result}"
        return await send_email("Task Reminder", f"Reminder: {task_description} due on {due}", GMAIL_RECIPIENT) + f"\n{sheets_result}"
    except Exception as e:
        logger.error(f"Error adding reminder: {e}")
        return f"Failed to add reminder: {str(e)}"

@mcp.tool()
async def delete_task(user_id: str, task_description: str) -> str:
    """Delete a task (database commented out, updates Sheets with empty list)."""
    try:
        # conn = get_db_connection()
        # cur = conn.cursor()
        # cur.execute("SELECT id, task_description FROM tasks WHERE user_id = %s AND task_description ILIKE %s", (user_id, f"%{task_description}%"))
        # task = cur.fetchone()
        # if not task:
        #     cur.close()
        #     conn.close()
        #     return f"No task found matching: {task_description}"
        # task_id = task[0]
        # cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
        # conn.commit()
        # cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
        # tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
        tasks = []  # No tasks since database is disabled
        sheets_result = await update_sheets(user_id, json.dumps(tasks))
        # cur.close()
        # conn.close()
        return f"Task '{task_description}' deleted (database disabled, cleared Sheets).\n{sheets_result}"
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return f"Failed to delete task: {str(e)}"

@mcp.tool()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def update_sheets(user_id: str, tasks_json: str) -> str:
    """Update Google Sheets with tasks."""
    if not sheets_service:
        return "Sheets service unavailable."
    try:
        tasks_list = json.loads(tasks_json)
        spreadsheet_id = SHEET_ID
        body = {'values': [[t['description'], t['due_date'], t['status'], t['priority']] for t in tasks_list]}
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1:D",
            valueInputOption="RAW",
            body=body
        ).execute()
        return "Spreadsheet updated successfully."
    except Exception as e:
        logger.error(f"Error updating Sheets: {e}")
        return f"Failed to update Sheets: {str(e)}"

@mcp.tool()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def retrieve_sheets_data(user_id: str) -> str:
    """Retrieve tasks from Google Sheets."""
    if not sheets_service:
        return "Sheets service unavailable."
    try:
        spreadsheet_id = SHEET_ID
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1:D"
        ).execute()
        values = result.get('values', [])
        if not values:
            return "No tasks found in the spreadsheet."
        formatted_tasks = "\n".join([f"Task: {row[0]}, Due: {row[1]}, Status: {row[2]}, Priority: {row[3]}" for row in values if len(row) >= 4])
        return f"Tasks in spreadsheet:\n{formatted_tasks}"
    except Exception as e:
        logger.error(f"Error retrieving Sheets data: {e}")
        return f"Failed to retrieve Sheets data: {str(e)}"

@mcp.tool()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def schedule_meeting(attendee_email: str, time_str: str, description: str = "Meeting") -> str:
    """Schedule a meeting using Google Calendar."""
    if not calendar_service:
        return "Calendar service unavailable."
    try:
        time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))|(\d{1,2}:\d{2}\s*hrs)", time_str, re.IGNORECASE)
        if not time_match:
            return "Invalid time format. Use '6:00 PM' or '18:00 hrs'."
        time_str = time_match.group(0).replace("hrs", "").strip()
        time = datetime.strptime(time_str, "%I:%M %p") if "AM" in time_str.upper() or "PM" in time_str.upper() else datetime.strptime(time_str, "%H:%M")
        time = time.replace(year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
        end_time = time + timedelta(hours=1)
        event = {
            'summary': description,
            'start': {'dateTime': time.isoformat(), 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
            'attendees': [{'email': attendee_email}],
        }
        event = calendar_service.events().insert(calendarId='primary', body=event).execute()
        return f"Meeting scheduled with {attendee_email} at {time_str}."
    except Exception as e:
        logger.error(f"Error scheduling meeting: {e}")
        return f"Failed to schedule meeting: {str(e)}"

# @mcp.tool()
# async def summarize_tasks(user_id: str) -> str:
#     """Summarize pending tasks."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#         tasks = cur.fetchall()
#         cur.close()
#         conn.close()
#         if not tasks:
#             return "No pending tasks found."
#         formatted_tasks = "\n".join([f"Task: {row[0]}, Due: {row[1].strftime('%Y-%m-%d')}, Status: {row[2]}, Priority: {row[3]}" for row in tasks])
#         return f"Pending tasks:\n{formatted_tasks}"
#     except Exception as e:
#         logger.error(f"Error summarizing tasks: {e}")
#         return f"Failed to summarize tasks: {str(e)}"

@mcp.tool()
async def rag_query(query: str) -> str:
    """Handle RAG queries with FAISS and LLM."""
    try:
        nodes = retriever.invoke(query)
        context = "\n".join([n.page_content for n in nodes])
        prompt = f"Using only the following context, answer '{query}':\n{context}\nIf the context doesn't fully answer, say so."
        response = rag_llm.invoke(prompt).content
        return response
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"RAG failed: {str(e)}"

# def send_daily_summary():
#     """Send daily summary email with tasks and emails (background task, not a tool)."""
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT user_id FROM user_profiles")
#         users = cur.fetchall()
#         for user_tuple in users:
#             user_id = user_tuple[0]
#             email_summary = asyncio.run(check_important_emails(user_id))
#             cur.execute("SELECT task_description, due_date, status, priority FROM tasks WHERE user_id = %s AND status = 'pending'", (user_id,))
#             tasks = [{'description': row[0], 'due_date': row[1].strftime('%Y-%m-%d'), 'status': row[2], 'priority': row[3]} for row in cur.fetchall()]
#             task_summary = "\n".join([f"{t['description']} (Due: {t['due_date']}, Priority: {t['priority']})" for t in tasks]) or "No pending tasks."
#             summary = f"Daily Summary (6 AM):\n\nImportant Emails:\n{email_summary}\n\nTasks:\n{task_summary}"
#             asyncio.run(send_email("Daily Summary", summary))
#             asyncio.run(update_sheets(user_id, json.dumps(tasks)))
#         cur.close()
#         conn.close()
#     except Exception as e:
#         logger.error(f"Daily summary error: {e}")

# # Start background scheduler for daily summary
# scheduler = BackgroundScheduler()
# scheduler.add_job(send_daily_summary, 'date', run_date=datetime.now() + timedelta(seconds=30))
# scheduler.start()

if __name__ == "__main__":
    mcp.run(transport="stdio")
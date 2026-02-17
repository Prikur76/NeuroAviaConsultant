import os
import re
import httpx
from datetime import datetime
from contextlib import asynccontextmanager
from environs import Env
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

env = Env()
env.read_env()


# --- Настройки и ключи ---
# Убедитесь, что переменная окружения PROXYAPI_API_KEY установлена
API_KEY = env.str("OPENAPI_API_KEY", "YOUR-KEY")
BASE_URL = env.str("OPENAPI_BASE_URL", "YOUR-URL-ENDPOINT")

# Глобальные переменные
total_requests_count = 0
vector_db = None


# --- Вспомогательные функции ---
def load_document_text(url: str) -> str:
    """Извлекает текст из Google Docs с поддержкой редиректов."""
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if not match:
        raise ValueError("Некорректная ссылка на Google Doc")
    
    doc_id = match.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    
    # Исправление: добавляем follow_redirects=True
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(export_url)
        response.raise_for_status()
        return response.text


# Описываем логику запуска и завершения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- [STARTUP] Код при запуске ---
    global vector_db
    doc_url = "https://docs.google.com/document/d/11MU3SnVbwL_rM-5fIC14Lc3XnbAV4rY1Zd_kpcMuH4Y"    
    print("🚀 Загрузка базы знаний...")
    try:
        raw_text = load_document_text(doc_url)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        source_chunks = splitter.create_documents([raw_text])        
        embeddings = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL)
        vector_db = FAISS.from_documents(source_chunks, embeddings)
        print("✅ База знаний успешно загружена и проиндексирована.")
    except Exception as e:
        print(f"❌ Ошибка при инициализации базы знаний: {e}")
    
    yield  # Здесь приложение начинает принимать запросы
    
    # --- [SHUTDOWN] Код при выключении ---
    print("🛑 Завершение работы приложения...")
    if vector_db:
        # В случае с FAISS в памяти очистка обычно не требуется, 
        # но здесь можно закрывать соединения с внешними БД.
        vector_db = None
    print("✅ Работа приложения завершена.")


app = FastAPI(
    title="Нейро-консультант: Авиационное страхование",
    description="API на базе правил страхования ответственности аэропортов",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    version="1.0.0"    
)

# --- Модели API ---
class QuestionRequest(BaseModel):
    question: str

# --- Эндпоинты ---

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Neuro Avia Consultant — Нейро-консультант</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body class="bg-slate-50 font-sans">
        <div class="min-h-screen flex flex-col items-center justify-center p-6">
            <div class="max-w-4xl w-full bg-white shadow-2xl rounded-3xl overflow-hidden flex flex-col md:flex-row">
                <div class="md:w-1/2 bg-indigo-700 p-10 text-white flex flex-col justify-between">
                    <div>
                        <div class="flex items-center space-x-3 mb-6">
                            <i class="fas fa-plane-departure text-3xl text-indigo-300"></i>
                            <h1 class="text-2xl font-bold tracking-tight uppercase">Neuro Avia Consultant</h1>
                        </div>
                        <p class="text-indigo-100 text-lg leading-relaxed">
                            Профессиональный ассистент по правилам страхования ответственности аэропортов.
                        </p>
                    </div>
                    <div class="mt-8 flex items-center space-x-2 text-xs font-mono text-indigo-300">
                        <span class="relative flex h-2 w-2">
                          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        <span>SERVER_STATUS: ONLINE</span>
                    </div>
                </div>

                <div class="md:w-1/2 p-10 flex flex-col justify-center">
                    <h2 class="text-slate-800 text-sm font-black uppercase tracking-widest mb-2 text-center md:text-left">Активность системы</h2>
                    
                    <div class="bg-slate-50 rounded-2xl p-8 mb-8 border border-slate-100 text-center md:text-left shadow-inner">
                        <div class="text-5xl font-black text-indigo-600 mb-1" id="request-counter">{total_requests_count}</div>
                        <p class="text-slate-400 text-sm font-medium uppercase tracking-tighter">Всего обработано вопросов</p>
                    </div>

                    <div class="grid gap-4">
                        <a href="/docs" class="bg-indigo-600 hover:bg-indigo-700 text-white text-center py-4 rounded-xl font-bold transition-all transform hover:-translate-y-1 shadow-lg shadow-indigo-100">
                            Запустить консоль (Swagger)
                        </a>
                        <a href="/stats" class="bg-white border border-slate-200 text-slate-600 text-center py-4 rounded-xl font-bold hover:bg-slate-50 transition-all">
                            Просмотреть JSON метрики
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Функция для обновления счетчика без перезагрузки страницы
            async function updateCounter() {{
                try {{
                    const response = await fetch('/stats');
                    const data = await response.json();
                    const counterElement = document.getElementById('request-counter');
                    
                    // Добавляем эффект плавного изменения, если число поменялось
                    if (counterElement.innerText != data.total_requests) {{
                        counterElement.style.transform = 'scale(1.1)';
                        counterElement.innerText = data.total_requests;
                        setTimeout(() => {{ counterElement.style.transform = 'scale(1)'; }}, 200);
                    }}
                }} catch (error) {{
                    console.error('Ошибка обновления статистики:', error);
                }}
            }}

            // Обновляем каждые 3 секунды
            setInterval(updateCounter, 3000);
        </script>
        
        <style>
            #request-counter {{ transition: transform 0.2s ease-in-out; }}
        </style>
    </body>
    </html>
    """

@app.post("/ask", summary="Задать вопрос консультанту", tags=["Консультация"])
async def ask_expert(request: QuestionRequest):
    """
    Метод обрабатывает вопрос, увеличивает счетчик и возвращает ответ эксперта.
    """
    global total_requests_count
    total_requests_count += 1
    
    if vector_db is None:
        raise HTTPException(status_code=503, detail="База знаний еще загружается")

    try:
        # Поиск контекста
        docs = vector_db.similarity_search(request.question, k=4)
        context = "\n\n".join([doc.page_content.strip() for doc in docs])

        # Системный промпт из вашего ДЗ
        system_prompt = (
            "Вы — сертифицированный эксперт по страхованию ответственности аэропортов. "
            "Говорите от первого лица как практикующий специалист. "
            "СТРОГО ЗАПРЕЩЕНО упоминать источники информации (никаких 'согласно документу' или 'в базе знаний'). "
            "Используйте ТОЛЬКО факты из контекста."
        )

        user_prompt = f"Контекст:\n{context}\n\nВопрос клиента:\n{request.question}"

        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        return {"answer": completion.choices[0].message.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/stats", summary="Получить статистику обращений", tags=["Инфо"])
async def get_stats():
    """
    Возвращает общее количество обращений всех пользователей.
    """
    return {
        "total_requests": total_requests_count,
        "timestamp": datetime.now().isoformat()
    }

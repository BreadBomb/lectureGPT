import asyncio
import os
import uuid
from asyncio import Queue

import socketio
from aiohttp import web
from langchain import LlamaCpp, OpenAI
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager, AsyncCallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader, MathpixPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings

from settings import Settings
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

from constants import CHROMA_SETTINGS
import signal
import sys

gpt_settings = Settings()
# create a Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")

loop = asyncio.new_event_loop()

# wrap with a WSGI application
app = web.Application()

qa = None

_status = {"model": "stopped"}

key = "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"

os.environ["OPENAI_API_KEY"] = key

q = Queue()

db: Chroma = None


def signal_handler(sig, frame):
    print("Shutdown...")
    del db
    app.shutdown()
    del sio
    del qa
    sys.exit(0)


class StreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs):
        global sio
        await sio.emit("chat_stream", token)


def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    global gpt_settings
    global q
    llm_type = gpt_settings.get_llm()

    if llm_type == "llama":
        local_llm = LlamaCpp(
            model_path="./model/llama-7b.ggmlv3.q4_0.bin",
            callback_manager=callback_manager,
            n_ctx=2048,
            use_mlock=True,
            verbose=True
        )
    else:
        local_llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", streaming=True,
                               callbacks=[StreamingCallbackHandler()],
                               temperature=0.7)

    return local_llm


async def start_model():
    global db
    if _status["model"] != "stopped":
        return
    _status["model"] = "starting"
    await sio.emit("status", _status)

    global gpt_settings
    llm_type = gpt_settings.get_llm()

    if llm_type == "llama":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                   model_kwargs={"device": "cpu"})
    else:
        embeddings = OpenAIEmbeddings(chunk_size=1)

    # load the vectorstore
    db = Chroma(collection_name="lectures",
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    _status["model"] = "started"
    await sio.emit("status", _status)


@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)


@sio.event
def status(sid, data):
    return _status


@sio.event
async def start(sid, data):
    await start_model()
    return None


@sio.event
async def chat(sid, key):
    global gpt_settings
    return gpt_settings.get_chat(key)


@sio.event
async def settings(sid, data):
    global gpt_settings
    return {
        "model": gpt_settings.get_llm(),
        "apiKey": key
    }


@sio.event
async def model(sid, model):
    global gpt_settings
    global key
    gpt_settings.set_llm(model)
    return {
        "model": gpt_settings.get_llm(),
        "apiKey": key
    }


@sio.event
async def documents(sid, scope):
    global db
    if db is None:
        return None
    result = db._collection.get(where={"scope": scope})
    return result


@sio.event
async def upload(sid, file, tags):
    global db
    file_uuid = uuid.uuid4()
    with open(f"tmp/{file_uuid}.pdf", "wb") as f:
        f.write(file)
        f.flush()
        f.close()
    document = PDFMinerLoader(f"tmp/{file_uuid}.pdf").load()[0]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents([document])
    for x in texts:
        x.metadata = tags
    db.add_documents(documents=texts)
    db.persist()
    return "ok"


@sio.event
async def query(sid, data):
    print("got query %s" % data)
    global qa
    global gpt_settings
    llm = gpt_settings.get_llm()
    gpt_settings.set_chat("all", data, "question")
    if qa is None:
        return
    try:
        if llm == "openai":
            result = await qa.acall(data)
        else:
            result = qa.run(data)
    except:
        return {"error": True, "message": ""}
    gpt_settings.set_chat("all", result["result"], "answer")
    return {"message": result["result"], "error": False}


if __name__ == "__main__":
    # main()
    signal.signal(signal.SIGINT, signal_handler)
    sio.attach(app)
    web.run_app(app)

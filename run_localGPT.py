import json

from aiohttp import web
from langchain import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from settings import Settings
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
import click

from constants import CHROMA_SETTINGS

import socketio

settings = Settings()
# create a Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")

# wrap with a WSGI application
app = web.Application()

qa = None

def load_model():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    local_llm = LlamaCpp(
        model_path="./model/llama-7b.ggmlv3.q4_0.bin",
        callback_manager=callback_manager,
        n_ctx=2048,
        use_mlock=True,
        verbose=True
    )

    return local_llm


def start_model():
    settings = Settings()
    settings.set_llm("openai")
    print(settings.get_llm())

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": "mps"})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    sio.emit("status", {"model": "running"})
    # # Interactive questions and answers
    # while True:
    #     query = input("\nEnter a query: ")
    #     if query == "exit":
    #         break
    #
    #     # Get the answer from the chain
    #     res = qa(query)
    #     answer, docs = res['result'], res['source_documents']
    #
    #     # Print the result
    #     print("\n\n> Question:")
    #     print(query)
    #     print("\n> Answer:")
    #     print(answer)
    #
    #     # # Print the relevant sources used for the answer
    #     print("----------------------------------SOURCE DOCUMENTS---------------------------")
    #     for document in docs:
    #         print("\n> " + document.metadata["source"] + ":")
    #         print(document.page_content)
    #     print("----------------------------------SOURCE DOCUMENTS---------------------------")


@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)


@sio.event
def status(sid, data):
    if qa is not None:
        return json.dumps({"model": "running"})
    if qa is None:
        return json.dumps({"model": "stopped"})


if __name__ == "__main__":
    # main()
    sio.attach(app)
    web.run_app(app)

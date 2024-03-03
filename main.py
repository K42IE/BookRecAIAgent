import click
import gradio as gr
from dotenv import load_dotenv
import os
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
import llama_index
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from pymilvus import connections
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import sys


def get_agent(verbose=False):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    load_dotenv()

    try:
        vector_store_books = MilvusVectorStore(dim=384, collection_name="books")
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/books",
            vector_store=vector_store_books,
        )
        books_index = load_index_from_storage(storage_context)
    except Exception as error:
        print(f'Unable to load index from storage: {error}')
        print('Indexing book dataset')
        vector_store_books = MilvusVectorStore(dim=384, collection_name="books", overwrite=True)
        book_docs = SimpleDirectoryReader(input_dir="./data").load_data()
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store_books,
        )
        books_index = VectorStoreIndex.from_documents(book_docs, storage_context=storage_context)
        books_index.storage_context.persist(persist_dir="./storage/books")

    books_query_engine = books_index.as_query_engine(similarity_top_k=3, verbose=verbose)

    tools = [
        QueryEngineTool(
            query_engine=books_query_engine,
            metadata=ToolMetadata(
                name="books_data",
                description="Provides information about known books; ONLY books known to this tool should be considered when answering questions about books",
            ),
        ),
    ]

    # # This is the main agent
    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=verbose, context=context)

    return agent

@click.group()
def cli():
    pass


@cli.command()
@click.option("--verbose/--no-verbose", default=False)
@click.option("--debug/--no-debug", default=False)
def repl(verbose: bool, debug: bool):
    if debug:
        llama_index.core.set_global_handler("simple")
        logging.basicConfig(stream=sys.stderr)
    get_agent(verbose).chat_repl()


@cli.command()
def gui():
    agent = get_agent(verbose=True)

    def chat_interface(prompt):
        # Send the prompt to the agent and get the response
        print(f'Got prompt: {prompt}')
        response = agent.chat(prompt)
        print(f'Got response: {response}')
        return response

    iface = gr.Interface(fn=chat_interface, 
                         inputs="text", 
                         outputs="text")

    iface.launch(share=True)

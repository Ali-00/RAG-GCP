# app/rag.py

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.storage import VECTORSTORE_DIR
from app.config import LLM_MODEL


def build_qa_chain():
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR.as_posix(),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below:

        Context:
        {context}

        Question:
        {question}
        """
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

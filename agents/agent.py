from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatGooglePalm,ChatOllama
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"]= os.getenv("GOOGLE_API_KEY")


#default tools
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki_tool.name)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

print(arxiv_tool.name)


loader=WebBaseLoader('https://docs.smith.langchain.com')
docs=loader.load()
chunks=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100).split_documents(docs)
vectorstore=FAISS.from_documents(chunks,HuggingFaceEmbeddings())
retriver= vectorstore.as_retriever()


from langchain.tools.retriever import create_retriever_tool
retriver_tool=create_retriever_tool(retriver,"langchain search",
                      "A tools that is say any details about lanchain")
print(retriver_tool.name)


tools=[wiki_tool,arxiv_tool,retriver_tool]



from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")



llm = ChatOllama(model="gemma:2b",temperature=0.2)


from langchain.agents import AgentExecutor, create_structured_chat_agent
agent = create_structured_chat_agent(llm=llm, tools=tools,prompt=prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
#print(agent_executor)

agent_executor.invoke({"input":"what is langsmith?"})
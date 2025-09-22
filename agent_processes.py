from multiprocessing import Process

def run_ingestion_agent(ing_in, ing_out):
    from ingestion_agent import IngestionAgent
    agent = IngestionAgent(ing_in, ing_out)
    while True:
        msg = ing_in.get()
        try:
            agent.run_once(msg)
        except Exception as e:
            ing_out.put({'type':'ERROR','sender':'IngestionAgent','payload':{'error':str(e)}})

def run_retrieval_agent(ret_in, ret_out, K_RETRIEVE, K_RERANK, embedding_model):
    from retrieval_agent import RetrievalAgent
    agent = RetrievalAgent(ret_in, ret_out, K_RETRIEVE=K_RETRIEVE, K_RERANK=K_RERANK, embedding_model=embedding_model)
    while True:
        msg = ret_in.get()
        try:
            agent.run_once(msg)
        except Exception as e:
            ret_out.put({'type':'ERROR','sender':'RetrievalAgent','payload':{'error':str(e)}})

def run_llm_agent(llm_in, llm_out, model_name="llama3.2:1b"):
    from llm_response_agent import LLMResponseAgent
    agent = LLMResponseAgent(llm_in, llm_out, model_name=model_name)
    while True:
        msg = llm_in.get()
        try:
            agent.run_once(msg)
        except Exception as e:
            llm_out.put({'type':'ERROR','sender':'LLMResponseAgent','payload':{'error':str(e)}})

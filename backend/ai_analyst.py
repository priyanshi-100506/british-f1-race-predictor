from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import os
import re

def ask_analyst(question: str):
    db = SQLDatabase.from_uri('postgresql://f1user:f1password@localhost:5432/f1analytics')
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=os.getenv('GROQ_API_KEY'))
    
    chain = create_sql_query_chain(llm, db)
    raw_response = chain.invoke({'question': question})
    
    # Extract SQL: Look for the SELECT statement and ignore conversational text
    sql_match = re.search(r"SELECT.*", raw_response, re.IGNORECASE | re.DOTALL)
    if sql_match:
        sql_query = sql_match.group(0)
    else:
        sql_query = raw_response  # Fallback
        
    print(f'AI generated SQL: {sql_query}')
    return db.run(sql_query)
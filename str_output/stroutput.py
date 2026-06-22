from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template = 'Write a Summary on the following {topic}',
    input_variables = ['topic']
)

prompt = template1.invoke({'topic': 'Outer Space'})

# chain = template1 | model | parser

result = model.invoke(prompt)

# result1 = chain.invoke({'topic' : 'Anime'})

print(result.content)

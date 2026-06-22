from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens=100,
    temperature=1.5,
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description = 'Name of the person')
    age: int = Field(gt = 18, description = 'Age of the person')
    city: str = Field(description = 'Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of a fictonal {place} person \n {format_instructions}',
    input_variables = ['place'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

# prompt = template.invoke({'place' : 'India'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'place' : 'America'})

print(final_result)
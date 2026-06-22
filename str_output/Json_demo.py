from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens=100,
    temperature=1.5,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'ive me the name, age and city of a fictional personG \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

# final_result = parser.parse(result.content)
# print(type(final_result))

print(result)
 
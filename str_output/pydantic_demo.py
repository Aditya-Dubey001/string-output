from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name: str = 'Aditya'
    age: Optional[int] = None
    

new_student = {'age' : 25}

student = Student(**new_student)

print(student)
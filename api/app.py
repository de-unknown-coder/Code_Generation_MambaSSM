from fastapi import FastAPI
from inference.generate import promptFormat, generate
from fastapi import HTTPException
from pydantic import BaseModel, field_validator

app= FastAPI(
    title="Mamba Code Generator API",
    description="Generate code using custom Mamba model",
    version="1.0.0"
)
class GenerateRequest(BaseModel):
    instruction: str
    input: str = "< noinput >"
    
    @field_validator("instruction")
    def validate_instruction(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Instruction cannot be empty or just spaces")

        if len(value) > 300:
            raise ValueError("Instruction must be at most 300 characters")

        return value
    @field_validator("input")
    def validate_input(cls, value):
        value = value.strip()

        if len(value) > 500:
            raise ValueError("Input must be at most 500 characters")

        return value
        
    

@app.post("/generate")
def generate_code(req : GenerateRequest):
    try:
        prompt = promptFormat(req.instruction, req.input)
        generated_code = generate(prompt)
        return {"generated_code": generated_code}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Model inference failed")
        print(e)
    
    

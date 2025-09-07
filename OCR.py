kk = "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"
import easyocr
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# 1. Init OCR
reader = easyocr.Reader(['en','hi'])

# 2. Read text from image
ocr_results = reader.readtext("/content/dolo.jpg")
ocr_texts = [text for _, text, conf in ocr_results if conf > 0.2]  # filter good confidence
ocr_input = " ".join(ocr_texts) if ocr_texts else "No text found"

# 3. Init LLM

llm = ChatGroq(
    groq_api_key=kk,
    model_name="openai/gpt-oss-20b",
    temperature=0.3
)

# 4. System Prompt
system_prompt = """You are an assistant that corrects OCR text from medicine labels.
The OCR text may be jumbled, incomplete, or misspelled.
Your task is:
1. Reconstruct the most likely correct medicine name.
2. Explain in 1 point what is the medicine is used for,the audience is layman use easy  terms."""

# 5. Create messages
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=ocr_input)
]

# 6. Get Response
response = llm(messages)
print("OCR Input:", ocr_input)
print("LLM Output:\n", response.content)

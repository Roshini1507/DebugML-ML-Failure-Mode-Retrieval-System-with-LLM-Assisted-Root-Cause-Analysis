import ollama

try:
    print("Testing Ollama connection...")

    response = ollama.chat(
        model="llama3:latest",
        messages=[
            {"role": "user", "content": "Reply ONLY with this JSON: {\"status\":\"ok\"}"}
        ]
    )

    print("SUCCESS:")
    print(response)

except Exception as e:
    print("FAILED:")
    print(str(e))

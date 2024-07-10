import openai

openai.api_key = "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"
def get_advice(question):
  completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a travel agent who is supposed to help clients with travel choices."},
      {"role": "user", "content": question}
    ]
  )

  return (completion.choices[0].message)

inp = "ask a question to traveling agent here:"
while True:
  question = input(inp)
  if question == "EXIT":
    break
  else:
    out = get_advice(question)
    inp = out
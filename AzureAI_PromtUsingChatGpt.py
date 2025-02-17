import openai
import os

# Seting Azure OpenAI API key and endpoint 
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE") 
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"

deployment_name = "gpt-4o-mini"

def generate_text(prompt, engine=deployment_name, max_tokens=150, temperature=0.7, n=1, stop=None):

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=stop,
    )

    return response.choices[0].text.strip()  # Returns the generated text

def main():

    # Basic Prompting:
    basic_prompt = "Write a short story about playing Tennis."
    basic_story = generate_text(basic_prompt)
    print("Basic Story:\n", basic_story)

    # Few-Shot Learning:
    few_shot_prompt = """
    Translate English to French:

    English: Hello, how are you?
    French: Bonjour, comment allez-vous?

    English: I would like a coffee.
    French: Je voudrais un café.

    English: Where is the library?
    French: Où est la bibliothèque?

    English: The book is on the table.
    French: 
    
    french_translation = generate_text(few_shot_prompt)
    print("\nFrench Translation:\n", french_translation)

    # Instruction Following:
    instruction_prompt = "Write a poem about moon light over ocean."
    poem = generate_text(instruction_prompt)
    print("\nPoem:\n", poem)

    # JSON Format:
    format_prompt = "Create a JSON object with the following keys: name, age, city.  Populate it with example data."
    json_output = generate_text(format_prompt)
    print("\nJSON Output:\n", json_output)

    # Temperature play around :
    creative_prompt = "Write a very creative and imaginative short story about a journey to Jupiter."
    creative_story = generate_text(creative_prompt, temperature=0.9) # Higher temp for more creativity
    print("\nCreative Story (Higher Temp):\n", creative_story)

    structured_prompt = "Give me a list of 3 common flowers."
    structured_list = generate_text(structured_prompt, temperature=0.2) # Lower temp for more structure
    print("\nStructured List (Lower Temp):\n", structured_list)


if __name__ == "__main__":
    main()

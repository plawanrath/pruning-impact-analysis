"""Chat template formatting for each model family."""


def format_prompt(item: dict, model_name: str, tokenizer) -> str:
    system_msg = (
        "Answer the following multiple choice question by responding "
        "with only the letter (A, B, or C) of your chosen answer."
    )
    letters = ["A", "B", "C"]
    choices = "\n".join(
        f"{letters[i]}) {item['answer_choices'][i]}"
        for i in range(len(item["answer_choices"]))
    )
    user_msg = (
        f"Context: {item['context']}\n\n"
        f"Question: {item['question']}\n\n"
        f"{choices}\n\n"
        f"Answer:"
    )

    # Gemma 2 does not support system role — prepend to user message
    if "gemma" in model_name.lower():
        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

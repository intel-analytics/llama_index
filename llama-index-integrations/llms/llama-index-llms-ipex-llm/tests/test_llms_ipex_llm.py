from llama_index.llms.ipex_llm import IpexLLM


def messages_to_prompt(messages):
    assert 0
    print(messages)
    template = """<s>[INST] <<SYS>>\n    \n<</SYS>>\n\n{query_str} [/INST]"""
    prompt = "<s>[INST] "
    for message in messages:
        if message.role == "system":
            prompt += f"<<SYS>>\n{message.content}\n<</SYS>>\n"
        elif message.role == "user":
            prompt += f"\n{message.content} "
        elif message.role == "assistant":
            prompt += f"\n{message.content} "

    # add final assistant prompt
    prompt += "[/INST]"

    return prompt


def completion_to_prompt(completion):
    print("completion_to_prompt")
    return f"<s>[INST] <<SYS>>\n    \n<</SYS>>\n\n{completion} [/INST]"


template = """<s>[INST] <<SYS>>\n    \n<</SYS>>\n\n{query_str} [/INST]"""


def test_text_complete_ipex_llm():
    model_name = "meta/Llama-2-7b-chat-hf"
    low_bit = "sym_int4"
    llm = IpexLLM.from_model_id(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=512,
        max_new_tokens=64,
        load_in_low_bit=low_bit,
        query_wrapper_prompt=template,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
    )

    res = llm.complete("What is AI")
    assert res is not None

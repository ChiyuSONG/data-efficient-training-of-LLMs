import gradio as gr
import copy
import time
from datetime import datetime
from argparse import ArgumentParser
import torch
from transformers import (
    GenerationConfig
)
import os
from inference import Assistant
with open("model_identity.txt", 'r', encoding='utf8') as f:
    model_identity = f.readlines()
    if model_identity:
        model_identity = model_identity[0].strip()
max_context_len = 2048



def count_history_len(history):
    sum = 0
    for turn in history:
        sum += len(turn[0])
        sum += len(turn[1])
    return sum

def get_bot_message(message, chat_history):
    context = {"messages": []}
    history = copy.deepcopy(chat_history)

    if len(message) > max_context_len:
        return None

    while (count_history_len(history) + len(message)) > max_context_len:
        history = history[1:]
        sys.update(value=f"Context longer than {max_context_len} is ignored.")

    for turn in chat_history:
        context["messages"].append({"role": "user", "content": turn[0]})
        context["messages"].append({"role": "assistant", "content": turn[1]})

    context["messages"].append({"role": "user", "content": message})

    responses, scores = assistant.inference([context])
    bot_message = responses[0]

    return bot_message

def respond(message, chat_history, request: gr.Request):
    sys.update(value=welcome)
    bot_message = get_bot_message(message, chat_history)
    if bot_message is None:
        sys.update(value=f"Your input message should no longer than {max_context_len}.")
        return message, chat_history
    time.sleep(0.05)
    chat_history.append((message, bot_message))
    ip = str(request.client)
    now = datetime.now()
    record_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    output_path = os.path.join("gradio_chats", ip + ".txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(record_time+"\n")
        f.write("用户： " + message + "\n")
        f.write("助手： " + bot_message + "\n")

    return "", chat_history

def start_new_line(message, chat_history):
    message += "\n"
    return message, chat_history



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="ChiyuSONG/data-efficient-training-of-LLMs-v1"
    )
    args = parser.parse_args()

    with torch.no_grad():
        assistant = Assistant(args.model_name_or_path)
        tokenizer = assistant.tokenizer
        config = GenerationConfig(
            max_new_tokens=max_context_len // 2,
            min_length=1,
            do_sample=False,
            temperature=0.2,
            top_k=40,
            top_p=0.6,
            repetition_penalty=1.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id,
                          tokenizer.eot_token_id, tokenizer.user_token_id, tokenizer.assistant_token_id],
        )
        assistant.config = config
        assistant.set_model_identity(model_identity)

    with gr.Blocks() as demo:
        welcome = "Welcome!"
        sys = gr.Markdown(value=welcome)
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Press Shift+Enter to start a new line...")
        sub = gr.Button("Submit")
        clear = gr.ClearButton([msg, chatbot])

        sub.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(start_new_line, [msg, chatbot], [msg, chatbot])

    demo.launch(share=True)

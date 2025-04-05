import gradio as gr
import requests
import json

def stream_model_response(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:7b",  # 请根据需要替换模型名称
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(url, json=payload, stream=True)
    output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in data:
                token = data["response"]
                output += token
                yield output  # 逐步返回累计输出
                if data.get("done", False):
                    break

with gr.Blocks() as demo:
    prompt = gr.Textbox(lines=3, label="请输入对话内容")
    response_box = gr.Markdown(label="模型回复")
    submit_btn = gr.Button("提交")
    # 直接在 click 事件中传入 stream=True（不要再链式调用 .stream()）
    submit_btn.click(fn=stream_model_response, inputs=prompt, outputs=response_box, stream=True)

if __name__ == "__main__":
    demo.launch()

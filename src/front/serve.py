import gradio as gr
import httpx

BACK = "http://backend:8000"

def list_groups():
    try:
        r = httpx.get(f"{BACK}/groups", timeout=10)
        return r.json().get("groups", [])
    except:
        return ["trash"]

def up_file(audio, group):
    if audio is None:
        return "Файл?"
    with open(audio, "rb") as f:
        out = httpx.post(f"{BACK}/audio", files={"file": f}, data={"group": group or "trash"}, timeout=600)
    if out.status_code == 200:
        return out.json().get("items", 0)
    return f"Err {out.status_code}"

def qa(q, group):
    if not q:
        return ""
    out = httpx.post(f"{BACK}/qa", json={"query": q, "target": group or "trash"}, timeout=120)
    if out.status_code == 200:
        return out.json()["answer"]
    return f"Err {out.status_code}"

with gr.Blocks() as demo:
    gr.Markdown("ASR + QA")
    with gr.Tab("ASR"):
        audio = gr.Audio(label="Звук", type="filepath")
        group = gr.Dropdown(label="группа", value="trash", choices=list_groups, allow_custom_value=True)
        btn1 = gr.Button("Залить")
        nout = gr.Textbox(label="Чанков")
        btn1.click(up_file, inputs=[audio, group], outputs=nout)
    with gr.Tab("Q&A"):
        qin = gr.Textbox(label="Вопрос")
        group2 = gr.Dropdown(label="группа", value="trash", choices=list_groups, allow_custom_value=True)
        btn2 = gr.Button("Спросить")
        ans = gr.Textbox(label="Ответ", lines=8)
        btn2.click(qa, inputs=[qin, group2], outputs=ans)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

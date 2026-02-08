import customtkinter as ctk
import torch
import threading
from diffusers import StableDiffusionPipeline
from PIL import Image
from tkinter import filedialog, messagebox

sd = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False
)

sd.enable_attention_slicing() 

print("Moving SD Pipeline to GPU...")
sd = sd.to("cuda")

app = ctk.CTk()
app.geometry("800x500")
app.title("Flap Toolkit")

cimg = None

def prog(pipe, step, timestep, callback_kwargs):
    percent = int(((step + 1) / 67) * 100)
    generatebtn.configure(text=f"{percent}%" if percent < 100 else "Generate!")
    latents = callback_kwargs["latents"]
    with torch.no_grad():
        image = latents[0].detach().cpu().permute(1, 2, 0).numpy()
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image * 255).astype("uint8")
        pil_image = Image.fromarray(image)
    preview_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(400, 400))
    img.configure(image=preview_img)
    img.image = preview_img
    app.update_idletasks()
    return callback_kwargs

def genworker():
    global cimg
    prompt = textbox.get("1.0", "end-1c")
    print(f"generating with prompt {prompt}")
    result = sd(prompt, num_inference_steps=67, callback_on_step_end=prog).images[0]
    imag = ctk.CTkImage(light_image=result, dark_image=result, size=(400, 400))
    img.configure(image=imag)
    img.image = imag
    cimg = result

def generate():
    threading.Thread(target=genworker).start()

def export():
    global cimg
    if cimg is None:
        messagebox.showwarning("error", "unable to save: no image!")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Image As"
    )
    if file_path:
        try:
            cimg.save(file_path)
            print(f"saved {file_path}")
        except Exception as e:
            messagebox.showerror("save error", f"unable to save: {e}")

generatebtn = ctk.CTkButton(app, text="Generate!", width=80, height=40, command=generate)
generatebtn.place(x=715, y=430)

exportbtn = ctk.CTkButton(app, text="Export", width=80, height=20, command=export)
exportbtn.place(x=715, y=475)

textbox = ctk.CTkTextbox(app, width=705, height=65)
textbox.place(x=5, y=430)

img = ctk.CTkLabel(app, text="", width=400, height=400, fg_color="gray20", corner_radius=7)
img.place(relx=0.5, rely=0.428, anchor="center")

app.resizable(False, False)
app.mainloop()
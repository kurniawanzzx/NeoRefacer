import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from refacer import Refacer
import argparse
import imageio
import numpy as np
from PIL import Image
import tempfile
import base64
import pyfiglet
import shutil
import time

# HAPUS: import ngrok - TIDAK AMAN

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("NeoRefacer") + "\033[0m")
print("\033[93m⚠️  PERINGATAN KEAMANAN: Gunakan hanya untuk konten yang Anda miliki haknya!\033[0m")

def cleanup_temp(folder_path):
    """Bersihkan folder temporary dengan aman"""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print("✓ Gradio cache cleared successfully.")
    except Exception as e:
        print(f"⚠️  Error: {e}")

# Setup temporary folder dengan aman
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
if os.path.exists("./tmp"):
    cleanup_temp(os.environ['GRADIO_TEMP_DIR'])
os.makedirs("./tmp", exist_ok=True)

# Parse arguments - HAPUS argumen ngrok
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=8)
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--colab_performance", default=False, action="store_true")
# HAPUS: parser.add_argument("--ngrok", type=str, default=None)
# HAPUS: parser.add_argument("--ngrok_region", type=str, default="us")
args = parser.parse_args()

# Validasi input untuk keamanan
def validate_file_path(filepath):
    """Validasi file path untuk mencegah path traversal"""
    if filepath is None:
        return None
    # Normalisasi path
    abs_path = os.path.abspath(filepath)
    # Cek apakah file berada di direktori yang diizinkan
    allowed_dirs = [os.path.abspath("./tmp"), os.path.abspath("./output")]
    if not any(abs_path.startswith(d) for d in allowed_dirs):
        # Jika bukan di tmp/output, copy ke tmp untuk keamanan
        safe_path = os.path.join("./tmp", os.path.basename(filepath))
        try:
            shutil.copy2(filepath, safe_path)
            return safe_path
        except:
            return filepath
    return filepath

# Initialize dengan aman
try:
    refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
    num_faces = args.max_num_faces
except Exception as e:
    print(f"❌ Error initializing Refacer: {e}")
    print("⚠️  Aplikasi mungkin tidak berfungsi dengan benar.")
    refacer = None
    num_faces = 8

def create_dummy_image():
    """Buat dummy image dengan aman"""
    try:
        dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
        dummy.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"⚠️  Error creating dummy image: {e}")
        return None

def run_image(*vars):
    """Jalankan image reface dengan aman"""
    if refacer is None:
        return None
    
    try:
        image_path = validate_file_path(vars[0])
        if image_path is None:
            return None
            
        origins = vars[1:(num_faces+1)]
        destinations = vars[(num_faces+1):(num_faces*2)+1]
        thresholds = vars[(num_faces*2)+1:-2]
        face_mode = vars[-2]
        partial_reface_ratio = vars[-1]

        disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
        multiple_faces_mode = (face_mode == "Multiple Faces")

        faces = []
        for k in range(num_faces):
            if destinations[k] is not None:
                dest_path = validate_file_path(destinations[k])
                origin_path = validate_file_path(origins[k]) if origins[k] is not None else None
                
                faces.append({
                    'origin': origin_path if not multiple_faces_mode else None,
                    'destination': dest_path,
                    'threshold': thresholds[k] if not multiple_faces_mode else 0.0
                })

        return refacer.reface_image(image_path, faces, disable_similarity=disable_similarity, 
                                    multiple_faces_mode=multiple_faces_mode, 
                                    partial_reface_ratio=partial_reface_ratio)
    except Exception as e:
        print(f"❌ Error in run_image: {e}")
        return None

def run(*vars):
    """Jalankan video/gif reface dengan aman"""
    if refacer is None:
        return None, None
    
    try:
        video_path = validate_file_path(vars[0])
        if video_path is None:
            return None, None
            
        origins = vars[1:(num_faces+1)]
        destinations = vars[(num_faces+1):(num_faces*2)+1]
        thresholds = vars[(num_faces*2)+1:-3]
        preview = vars[-3]
        face_mode = vars[-2]
        partial_reface_ratio = vars[-1]

        disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
        multiple_faces_mode = (face_mode == "Multiple Faces")

        faces = []
        for k in range(num_faces):
            if destinations[k] is not None:
                dest_path = validate_file_path(destinations[k])
                origin_path = validate_file_path(origins[k]) if origins[k] is not None else None
                
                faces.append({
                    'origin': origin_path if not multiple_faces_mode else None,
                    'destination': dest_path,
                    'threshold': thresholds[k] if not multiple_faces_mode else 0.0
                })

        mp4_path, gif_path = refacer.reface(video_path, faces, preview=preview, 
                                            disable_similarity=disable_similarity, 
                                            multiple_faces_mode=multiple_faces_mode, 
                                            partial_reface_ratio=partial_reface_ratio)
        return mp4_path, gif_path if gif_path else None
    except Exception as e:
        print(f"❌ Error in run: {e}")
        return None, None

def load_first_frame(filepath):
    """Load frame pertama dengan aman"""
    if filepath is None:
        return None
    try:
        filepath = validate_file_path(filepath)
        frames = imageio.get_reader(filepath)
        return frames.get_data(0)
    except Exception as e:
        print(f"⚠️  Error loading first frame: {e}")
        return None

def extract_faces_auto(filepath, refacer_instance, max_faces=5, isvideo=False):
    """Ekstrak faces dengan aman"""
    if filepath is None or refacer_instance is None:
        return [None] * max_faces

    try:
        filepath = validate_file_path(filepath)
        
        if isvideo and os.path.getsize(filepath) > 5 * 1024 * 1024:
            print("Video too large for auto-extract, skipping face extraction.")
            return [None] * max_faces

        frame = load_first_frame(filepath)
        if frame is None:
            return [None] * max_faces

        while len(frame.shape) > 3:
            frame = frame[0]

        if frame.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 (RGB), but got {frame.shape[-1]}")

        temp_image_path = os.path.join("./tmp", f"temp_face_extract_{int(time.time() * 1000)}.png")
        Image.fromarray(frame).save(temp_image_path)

        try:
            faces = refacer_instance.extract_faces_from_image(temp_image_path, max_faces=max_faces)
            return faces + [None] * (max_faces - len(faces))
        finally:
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_image_path}: {e}")
    except Exception as e:
        print(f"⚠️  Error extracting faces: {e}")
        return [None] * max_faces

def toggle_tabs_and_faces(mode, face_tabs, origin_faces):
    """Toggle tabs berdasarkan mode"""
    if mode == "Single Face":
        tab_updates = [gr.update(visible=(i == 0)) for i in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    elif mode == "Multiple Faces":
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    else:
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=True) for _ in range(len(origin_faces))]
    return tab_updates + origin_updates
    
def handle_tif_preview(filepath):
    """Handle TIF preview dengan aman"""
    if filepath is None:
        return None
    try:
        filepath = validate_file_path(filepath)
        preview_path = os.path.join("./tmp", f"tif_preview_{int(time.time() * 1000)}.jpg")
        Image.open(filepath).convert('RGB').save(preview_path)
        return preview_path
    except Exception as e:
        print(f"⚠️  Error creating TIF preview: {e}")
        return None

# --- UI ---
theme = gr.themes.Base(primary_hue="blue", secondary_hue="cyan")

with gr.Blocks(theme=theme, title="NeoRefacer - AI Refacer") as demo:
    # Cek apakah icon ada
    icon_path = "icon.png"
    icon_html = ''
    if os.path.exists(icon_path):
        try:
            with open(icon_path, "rb") as f:
                icon_data = base64.b64encode(f.read()).decode()
            icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:40px;height:40px;margin-right:10px;">'
        except:
            icon_html = ''

    with gr.Row():
        gr.Markdown(f"""
        <div style="display: flex; align-items: center;">
        {icon_html}
        <span style="font-size: 2em; font-weight: bold; color:#2563eb;">NeoRefacer</span>
        </div>
        <div style="margin-top: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px;">
        ⚠️ <strong>Peringatan Keamanan:</strong> Gunakan hanya untuk konten yang Anda miliki haknya. 
        File hasil akan otomatis dihapus setelah sesi berakhir.
        </div>
        """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        with gr.Row():
            image_input = gr.Image(label="Original image", type="filepath")
            image_output = gr.Image(label="Refaced image", interactive=False, type="filepath")

        with gr.Row():
            face_mode_image = gr.Radio(["Single Face", "Multiple Faces", "Faces By Match"], value="Single Face", label="Replacement Mode")
            partial_reface_ratio_image = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            image_btn = gr.Button("Reface Image", variant="primary")

        origin_image, destination_image, thresholds_image, face_tabs_image = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_image.append(origin)
            destination_image.append(destination)
            thresholds_image.append(threshold)
            face_tabs_image.append(tab)

        face_mode_image.change(fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_image, origin_image), 
                              inputs=[face_mode_image], outputs=face_tabs_image + origin_image)
        demo.load(fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_image, origin_image), 
                 inputs=None, outputs=face_tabs_image + origin_image)

        if refacer:
            image_btn.click(fn=run_image, 
                           inputs=[image_input] + origin_image + destination_image + thresholds_image + [face_mode_image, partial_reface_ratio_image], 
                           outputs=[image_output])
            image_input.change(fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces), 
                              inputs=image_input, outputs=origin_image)
        else:
            gr.Markdown("⚠️ Refacer failed to initialize. Please check the logs.")
            
        image_input.change(fn=lambda _: 0.0, inputs=image_input, outputs=partial_reface_ratio_image)

    # --- GIF MODE ---
    with gr.Tab("GIF Mode"):
        with gr.Row():
            gif_input = gr.File(label="Original GIF", file_types=[".gif"])
            gif_preview = gr.Video(label="GIF Preview", interactive=False)
            gif_output = gr.Video(label="Refaced GIF (MP4)", interactive=False, format="mp4")
            gif_file_output = gr.Image(label="Refaced GIF (GIF)", type="filepath")

        with gr.Row():
            face_mode_gif = gr.Radio(["Single Face", "Multiple Faces", "Faces By Match"], value="Single Face", label="Replacement Mode")
            partial_reface_ratio_gif = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            gif_btn = gr.Button("Reface GIF", variant="primary")
            preview_checkbox_gif = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)

        origin_gif, destination_gif, thresholds_gif, face_tabs_gif = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_gif.append(origin)
            destination_gif.append(destination)
            thresholds_gif.append(threshold)
            face_tabs_gif.append(tab)

        face_mode_gif.change(fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_gif, origin_gif), 
                            inputs=[face_mode_gif], outputs=face_tabs_gif + origin_gif)
        demo.load(fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_gif, origin_gif), 
                 inputs=None, outputs=face_tabs_gif + origin_gif)

        if refacer:
            gif_btn.click(fn=run, 
                          inputs=[gif_input] + origin_gif + destination_gif + thresholds_gif + [preview_checkbox_gif, face_mode_gif, partial_reface_ratio_gif], 
                          outputs=[gif_output, gif_file_output])
            gif_input.change(fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces), 
                            inputs=gif_input, outputs=origin_gif)
        else:
            gr.Markdown("⚠️ Refacer failed to initialize. Please check the logs.")
            
        gif_input.change(fn=lambda file: file, inputs=gif_input, outputs=[gif_preview])
        gif_input.change(fn=lambda _: 0.0, inputs=gif_input, outputs=partial_reface_ratio_gif)

    # --- TIF MODE ---
    with gr.Tab("TIFF Mode"):
        with gr.Row():
            tif_input = gr.File(label="Original TIF", file_types=[".tif", ".tiff"])
            tif_preview = gr.Image(label="TIF Preview (Cover Page)", type="filepath")
            tif_output_preview = gr.Image(label="Refaced TIF Preview (Cover Page)", type="filepath")
            tif_output_file = gr.File(label="Refaced TIF (Download)", interactive=False)

        with gr.Row():
            face_mode_tif = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            partial_reface_ratio_tif = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            tif_btn = gr.Button("Reface TIF", variant="primary")

        origin_tif, destination_tif, thresholds_tif, face_tabs_tif = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_tif.append(origin)
            destination_tif.append(destination)
            thresholds_tif.append(threshold)
            face_tabs_tif.append(tab)

        face_mode_tif.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_tif, origin_tif),
            inputs=[face_mode_tif],
            outputs=face_tabs_tif + origin_tif
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_tif, origin_tif),
            inputs=None,
            outputs=face_tabs_tif + origin_tif
        )

        def process_tif(tif_path, *vars):
            """Process TIF dengan aman"""
            if tif_path is None:
                return None, None, None
            try:
                tif_path = validate_file_path(tif_path)
                
                original_img = Image.open(tif_path)
                if hasattr(original_img, "n_frames") and original_img.n_frames > 1:
                    original_img.seek(0)
                temp_preview_path = os.path.join("./tmp", f"tif_preview_{int(time.time() * 1000)}.jpg")
                original_img.convert('RGB').save(temp_preview_path)

                refaced_path = run_image(tif_path, *vars)
                if refaced_path is None:
                    return temp_preview_path, None, None

                refaced_img = Image.open(refaced_path)
                if hasattr(refaced_img, "n_frames") and refaced_img.n_frames > 1:
                    refaced_img.seek(0)
                temp_refaced_preview_path = os.path.join("./tmp", f"refaced_tif_preview_{int(time.time() * 1000)}.jpg")
                refaced_img.convert('RGB').save(temp_refaced_preview_path)

                return temp_preview_path, temp_refaced_preview_path, refaced_path
            except Exception as e:
                print(f"❌ Error processing TIF: {e}")
                return None, None, None

        if refacer:
            tif_btn.click(
                fn=lambda tif_path, *args: process_tif(tif_path, *args),
                inputs=[tif_input] + origin_tif + destination_tif + thresholds_tif + [face_mode_tif, partial_reface_ratio_tif],
                outputs=[tif_preview, tif_output_preview, tif_output_file]
            )
            tif_input.change(
                fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces),
                inputs=tif_input,
                outputs=origin_tif
            )
        else:
            gr.Markdown("⚠️ Refacer failed to initialize. Please check the logs.")
            
        tif_input.change(
            fn=handle_tif_preview,
            inputs=tif_input,
            outputs=tif_preview
        )
        tif_input.change(fn=lambda _: 0.0, inputs=tif_input, outputs=partial_reface_ratio_tif)

    # --- VIDEO MODE ---
    with gr.Tab("Video Mode"):
        with gr.Row():
            video_input = gr.Video(label="Original video", format="mp4")
            video_output = gr.Video(label="Refaced Video", interactive=False, format="mp4")

        with gr.Row():
            face_mode_video = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            partial_reface_ratio_video = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            video_btn = gr.Button("Reface Video", variant="primary")

        preview_checkbox_video = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)

        origin_video, destination_video, thresholds_video, face_tabs_video = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_video.append(origin)
            destination_video.append(destination)
            thresholds_video.append(threshold)
            face_tabs_video.append(tab)

        face_mode_video.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_video, origin_video),
            inputs=[face_mode_video],
            outputs=face_tabs_video + origin_video
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_video, origin_video),
            inputs=None,
            outputs=face_tabs_video + origin_video
        )
        
        if refacer:
            video_input.change(
                fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces, isvideo=True),
                inputs=video_input,
                outputs=origin_video
            )
        else:
            gr.Markdown("⚠️ Refacer failed to initialize. Please check the logs.")
        
        video_input.change(fn=lambda _: 0.0, inputs=video_input, outputs=partial_reface_ratio_video)

        if refacer:
            video_btn.click(
                fn=lambda *args: run(*args),
                inputs=[video_input] + origin_video + destination_video + thresholds_video + [preview_checkbox_video, face_mode_video, partial_reface_ratio_video],
                outputs=[video_output, gr.File(visible=False)]
            )

# HAPUS SEMUA KODE NGRIK
# if args.ngrok: ... (HAPUS BLOK INI)

# Tambahkan fungsi cleanup otomatis
def cleanup_on_exit():
    """Bersihkan file temporary saat aplikasi ditutup"""
    print("\n🧹 Membersihkan file temporary...")
    cleanup_temp("./tmp")
    # Hapus file output yang lebih dari 1 jam
    output_dir = "./output"
    if os.path.exists(output_dir):
        current_time = time.time()
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if current_time - os.path.getmtime(file_path) > 3600:  # 1 jam
                    try:
                        os.remove(file_path)
                        print(f"✓ Menghapus file lama: {file_path}")
                    except:
                        pass

# Daftarkan cleanup
import atexit
atexit.register(cleanup_on_exit)

# --- Launch app dengan aman ---
if __name__ == "__main__":
    print("\n🚀 Meluncurkan NeoRefacer...")
    print(f"🌐 Server akan berjalan di: http://{args.server_name}:{args.server_port}")
    if args.share_gradio:
        print("📢 Mode share aktif - Aplikasi akan dapat diakses publik melalui Gradio share link")
        print("⚠️  PERINGATAN: Jangan bagikan link ke orang yang tidak dikenal!")
    
    demo.queue().launch(
        favicon_path="icon.png" if os.path.exists("icon.png") else None,
        show_error=True, 
        share=args.share_gradio, 
        server_name=args.server_name, 
        server_port=args.server_port
    )

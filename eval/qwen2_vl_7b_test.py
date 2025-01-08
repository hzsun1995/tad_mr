import cv2
import json
import os
import re
import shutil
import tempfile
from tqdm import tqdm
import torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import argparse

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = '12345'




import torch.distributed as dist

dist.init_process_group(backend='nccl')



model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/media/junhuan/f735e2d6-fcde-4053-9be9-15ec61577e6e1/models/qwen2-vl-int4/", torch_dtype="auto", device_map="auto"
)


model = torch.nn.parallel.DistributedDataParallel(model)

processor = AutoProcessor.from_pretrained("/media/junhuan/f735e2d6-fcde-4053-9be9-15ec61577e6e1/models/qwen2-vl-int4/")

def annotate_frame_with_pil(frame, text, position, font_size, color):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    
    width, height = frame.size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    margin = 0
    if position == "top_left":
        x, y = margin, margin
    elif position == "top_right":
        x, y = width - text_width - margin, margin
    elif position == "bottom_left":
        x, y = margin, height - text_height - margin
    elif position == "bottom_right":
        x, y = width - text_width - margin, height - text_height - margin
    elif position == "center":
        x, y = (width - text_width) // 2, (height - text_height) // 2
    else:
        raise ValueError("Invalid position argument")

    if position in ["bottom_left", "bottom_right"]:
        y -= text_height / 3

    draw.text((x, y), text, font=font, fill=color)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame

def annotate_and_save_video(file_path, output_file_path, position, font_size, color):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {file_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = 1
        frame_interval = int(fps / target_fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, target_fps, (336, 336))

        frame_count = 0
        water_mark_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (336, 336))
                frame = annotate_frame_with_pil(frame, str(water_mark_count), position, font_size, color)
                out.write(frame)

                water_mark_count += 1
                
            frame_count += 1

        cap.release()
        out.release()

    except Exception as e:
        print(f"Error processing video {file_path}: {e}")


def process_video_query(model, processor, query, input_format, instruction, device="cuda", video_path=None, position='top_right', font_size=80, color='red'):
    temp_dir = tempfile.mkdtemp()
    try:
        video_file_path = video_path
        annotated_video_path = os.path.join(temp_dir, 'annotated_video.mp4')
        annotate_and_save_video(
            video_file_path,
            annotated_video_path,
            position=position,
            font_size=font_size,
            color=color
        )

        input_context = instruction + input_format.format(query)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": annotated_video_path,
                        "fps": 1
                    },
                    {"type": "text", "text": input_context},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.module.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0]
        
        match = re.search(r"from\s*(?:frame\s*)?(\d+)\s*to\s*(?:frame\s*)?(\d+)", response, re.IGNORECASE)
        if match:
            pred_start = int(match.group(1))
            pred_end = int(match.group(2))
            print(f"Query: {query}")
            print(f"Response: {response}")
            print(f"Predicted Start Frame: {pred_start}")
            print(f"Predicted End Frame: {pred_end}")
        else:
            print(f"Query: {query}")
            print(f"Response: {response}")
            print("No frame range found in the response.")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", 
                        default="The following input consists of video frames, with one frame per second, for a total of only 150 seconds.",
                        type=str, required=False, help="User input query.")

    parser.add_argument("--video_path",
                        default="/home/sunhaozhou/NumPro/test_video/qv_test/S73Z-nM0GQE_60.0_210.0.mp4",
                        type=str, required=False, help="Path to the input video file.")


    parser.add_argument("--input_format", type=str, default="During which frames can we see \"Woman holds up a small blue bottle.\"? Answer in the format of 'from x to y'.", help="Input format string for the query.")
    parser.add_argument("--instruction", type=str, default="The red numbers on each frame represent the frame number.", help="Instruction for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--position", type=str, default="bottom_right", help="Position of the frame number annotation.")
    parser.add_argument("--font_size", type=int, default=40, help="Font size of the frame number annotation.")
    parser.add_argument("--color", type=str, default="red", help="Color of the frame number annotation.")
    args = parser.parse_args()
    
    process_video_query(model, processor, args.query, args.input_format, args.instruction, args.device, args.video_path, args.position, args.font_size, args.color)
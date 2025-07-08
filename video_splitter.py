import cv2
import os

def split_video(input_video, output_dir, chunk_duration=20):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"[INFO] Video: {input_video}")
    print(f"[INFO] FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    chunk_frames = int(chunk_duration * fps)
    basename = os.path.splitext(os.path.basename(input_video))[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    chunk_idx = 0
    frame_idx = 0
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start a new chunk every chunk_frames
        if frame_idx % chunk_frames == 0:
            if out is not None:
                out.release()
            chunk_start = int(frame_idx / fps)
            chunk_end = int(min((frame_idx + chunk_frames) / fps, duration))
            chunk_filename = f"{basename}_chunk_{chunk_idx:03d}_{chunk_start:05d}-{chunk_end:05d}.mp4"
            chunk_path = os.path.join(output_dir, chunk_filename)
            out = cv2.VideoWriter(chunk_path, fourcc, int(fps), (frame.shape[1], frame.shape[0]))
            print(f"[INFO] Writing chunk: {chunk_filename}")
            chunk_idx += 1
            written = 0

        out.write(frame)
        written += 1
        frame_idx += 1

        # If we've written enough frames for this chunk, close it
        if written == chunk_frames:
            out.release()
            out = None

    if out is not None:
        out.release()
    cap.release()
    print(f"[INFO] Done! {chunk_idx} video chunks saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a video into N-second chunks.")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output_dir", required=True, help="Directory to save video chunks")
    parser.add_argument("--chunk_duration", type=int, default=20, help="Chunk duration in seconds (default: 20)")

    args = parser.parse_args()

    split_video(args.video, args.output_dir, chunk_duration=args.chunk_duration)
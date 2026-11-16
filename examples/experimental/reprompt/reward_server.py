"""The image half of RePrompt's reward as an HTTP service: render a prompt with Z-Image-Turbo the way RePrompt's
`PromptScorer` does, and score the image with ImageReward against the user's original prompt.

It runs outside Miles' environment, since ImageReward needs transformers 4.x: an env with `image-reward==1.5`,
`diffusers>=0.36` and `torch` is enough. One process serves one GPU and renders one image at a time, as each of
RePrompt's TRL processes did.

    CUDA_VISIBLE_DEVICES=2 python examples/experimental/reprompt/reward_server.py --port 18002

POST /score {"prompt": <text to render>, "ori_prompt": <user prompt>} answers {"image_reward": <float>}.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ImageReward as RM
import torch
from diffusers import ZImagePipeline


class _Scorer:
    def __init__(self, model: str):
        self.pipe = ZImagePipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.image_reward = RM.load("ImageReward-v1.0", device="cuda")
        self.lock = threading.Lock()

    def score(self, prompt: str, ori_prompt: str) -> float:
        with self.lock, torch.no_grad():
            # RePrompt's own call for this generator at its 512x512 resolution; Z-Image-Turbo is distilled to 9 steps
            # without CFG.
            image = self.pipe(
                prompt,
                num_images_per_prompt=1,
                height=512,
                width=512,
                guidance_scale=0.0,
                num_inference_steps=9,
                max_sequence_length=512,
            ).images[0]
            return float(self.image_reward.score(ori_prompt, [image]))


def _handler(scorer: _Scorer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/score":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps({"image_reward": scorer.score(body["prompt"], body["ori_prompt"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _handler(_Scorer(args.model)))
    print(f"reward server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

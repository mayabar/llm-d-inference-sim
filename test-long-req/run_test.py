import argparse
import random
import string
import time
import requests
import concurrent.futures

model = "HuggingFaceTB/SmolLM2-135M-Instruct"

def _word_stream():
    punctuation = [' ', ', ', '. ']
    weights = [85, 10, 5]
    while True:
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
        trail = random.choices(punctuation, weights=weights, k=1)[0]
        yield word + trail

def generate_text_by_tokens(num_tokens):
    parts = []
    stream = _word_stream()
    for _ in range(num_tokens):
        parts.append(next(stream))
    return ''.join(parts).rstrip(', ')

def generate_text_by_chars(num_chars):
    parts = []
    total = 0
    for chunk in _word_stream():
        parts.append(chunk)
        total += len(chunk)
        if total >= num_chars:
            break
    text = ''.join(parts)[:num_chars]
    if text and text[-1] in (',', '.'):
        text = text[:-1] + ' '
    return text

def make_prompt(args):
    if args.prompt_chars is not None:
        return generate_text_by_chars(args.prompt_chars)
    return generate_text_by_tokens(args.prompt_tokens)

def send_request(request_id, prompt_text):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 100,
    }

    print(f"[{request_id}] Sending prompt ({len(prompt_text)} chars) to {url}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"[{request_id}] Success!")
    except requests.exceptions.ConnectionError:
        print(f"[{request_id}] Error: Could not connect to {url}. Is your server running?")
    except requests.exceptions.RequestException as e:
        print(f"[{request_id}] HTTP Request failed: {e}")
        if response is not None:
            print(f"[{request_id}] Response content:", response.text)


def run_parallel(args):
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(send_request, i, make_prompt(args)) for i in range(args.parallel)]
        concurrent.futures.wait(futures)


def run_rate(args):
    interval = 1.0 / args.rate
    end_time = time.time() + args.duration
    request_id = 0
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = []
    while time.time() < end_time:
        tick = time.time()
        futures.append(executor.submit(send_request, request_id, make_prompt(args)))
        request_id += 1
        elapsed = time.time() - tick
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    concurrent.futures.wait(futures)
    executor.shutdown(wait=False)
    print(f"Sent {request_id} requests over {args.duration}s at {args.rate} req/s")


def main():
    parser = argparse.ArgumentParser(description="Send chat completion requests.")

    prompt_size = parser.add_mutually_exclusive_group(required=True)
    prompt_size.add_argument("--prompt-tokens", type=int, help="Prompt size as number of tokens (words)")
    prompt_size.add_argument("--prompt-chars", type=int, help="Prompt size as number of characters")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parallel", type=int, help="Mode 1: number of requests to send in parallel")
    mode.add_argument("--rate", type=float, help="Mode 2: requests per second")

    parser.add_argument("--duration", type=int, help="Duration in seconds (required with --rate)")
    args = parser.parse_args()

    if args.rate is not None and args.duration is None:
        parser.error("--duration is required when using --rate")

    if args.parallel is not None:
        run_parallel(args)
    else:
        run_rate(args)

if __name__ == "__main__":
    main()

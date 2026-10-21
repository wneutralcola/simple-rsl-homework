"""Collect real inference and GPU evidence; no third-party Python dependencies."""
import argparse
import datetime
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def api(host, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(host.rstrip('/') + endpoint, data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 'nvidia-smi unavailable; check loaded_models.size_vram instead.'
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='qwen3:4b-q4_K_M')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results' / 'deployment.json')
    args = parser.parse_args()
    started = time.perf_counter()
    prompt = '请用中文一句话说明什么是递归自学习。'
    response = api(args.host, '/api/chat', {
        'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
        'stream': False, 'think': False, 'keep_alive': '10m',
        'options': {'num_ctx': 4096, 'num_predict': 256, 'temperature': 0, 'seed': 42}
    })
    duration = time.perf_counter() - started
    ns = response.get('eval_duration', 0)
    result = {
        'time_china': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'host': args.host, 'model': args.model, 'version': api(args.host, '/api/version'),
        'gpu': gpu(), 'loaded_models': api(args.host, '/api/ps'),
        'prompt': prompt, 'response': response, 'wall_seconds': round(duration, 3),
        'output_tokens_per_second': round(response.get('eval_count', 0) * 1e9 / ns, 2) if ns else None,
    }
    if not response.get('done') or not response.get('message', {}).get('content', '').strip():
        raise RuntimeError('Inference did not finish with non-empty output.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import sys, json, base64
for line in sys.stdin:
    if line.startswith('data:'):
        try:
            d = json.loads(line[5:])
            if h := d.get('wav_header'):
                sys.stdout.buffer.write(base64.b64decode(h))
            elif c := d.get('pcm_chunk'):
                sys.stdout.buffer.write(base64.b64decode(c))
        except: pass

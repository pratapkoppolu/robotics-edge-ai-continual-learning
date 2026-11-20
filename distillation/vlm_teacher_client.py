"""A2: Gemini REST VLM adapter. Predictions are unapproved until A3 review."""
import base64
import json
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def parse_teacher_json(content):
    """Retain the approved candidate schema expected by existing A3 tools."""
    if not isinstance(content, str):
        raise ValueError("Teacher output must be a JSON string")
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    result = json.loads(content)
    if not isinstance(result, dict) or not isinstance(result.get('objects'), list):
        raise ValueError('Teacher must return a JSON object containing objects[]')
    objects = []
    for obj in result['objects']:
        if not isinstance(obj, dict):
            raise ValueError('Each object must be a dict')
        cls = obj.get('class_id')
        coords = obj.get('bbox_xyxy_normalized')
        if type(cls) is not int or not isinstance(coords, list) or len(coords) != 4:
            raise ValueError('Invalid class_id or bbox_xyxy_normalized')
        if not all(type(v) in (int, float) and 0 <= v <= 1 for v in coords):
            raise ValueError('Coordinates must be numeric values in [0,1]')
        x1, y1, x2, y2 = coords
        if not (x1 < x2 and y1 < y2):
            raise ValueError('Degenerate normalized box')
        objects.append({'class_id': cls, 'bbox_xyxy_normalized': coords})
    return {'objects': objects}


def invoke_vlm(image_path, cfg, task):
    """Send one actual image to Gemini; output remains a candidate, not an approved label."""
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY is missing in this Python process')
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    allowed_classes = cfg['task_classes'][task]
    prompt = (
        'Locate only visible objects relevant to the task. Return exactly one JSON object '
        'with key "objects" (array). Each object must have integer "class_id" and '
        '"bbox_xyxy_normalized" [x1,y1,x2,y2] where values are between 0 and 1. '
        'Class IDs and meanings: ' + json.dumps(allowed_classes) + '. '
        'Do not infer invisible objects. If none, return {"objects": []}. '
        'No other keys. Task description: ' + cfg['task_descriptions'][task]
    )
    mime = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
    if mime not in ('image/jpeg', 'image/png', 'image/webp'):
        raise ValueError(f'Unsupported image MIME type: {mime}')
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    payload = {
        'contents': [{'role': 'user', 'parts': [
            {'text': prompt},
            {'inline_data': {'mime_type': mime, 'data': encoded}},
        ]}],
        'generationConfig': {
            'temperature': 0,
            'maxOutputTokens': 1024,
            'responseMimeType': 'application/json',
        },
    }
    model = cfg['model_id']
    base = cfg['api_base'].rstrip('/')
    url = f'{base}/models/{quote(model, safe="")}:generateContent'
    request = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': key},
        method='POST',
    )
    try:
        with urlopen(request, timeout=int(cfg.get('timeout_seconds', 120))) as response:
            result = json.load(response)
    except HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace')
        # Do not log request headers or tokens.
        raise RuntimeError(f'Gemini API HTTP {exc.code}: {error_body[:1200]}') from exc
    except URLError as exc:
        raise RuntimeError(f'Cannot connect to Gemini API: {exc.reason}') from exc

    try:
        candidates = result['candidates']
        parts = candidates[0]['content']['parts']
        message = ''.join(part.get('text', '') for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Gemini did not return text in candidates[0].content.parts') from exc
    if not message.strip():
        raise ValueError(f'Gemini returned empty text; finishReason={candidates[0].get("finishReason")}')
    prediction = parse_teacher_json(message)
    class_ids = {int(v) for v in allowed_classes.keys()} if isinstance(allowed_classes, dict) and all(str(k).isdigit() for k in allowed_classes) else None
    if class_ids is not None:
        for obj in prediction['objects']:
            if obj['class_id'] not in class_ids:
                raise ValueError(f'Unrecognized teacher class_id: {obj["class_id"]}')
    return prediction

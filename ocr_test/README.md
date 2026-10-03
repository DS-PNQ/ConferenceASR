# OCR engine test (for the OnSpeak47 phone OCR)

Desktop test bench used to choose the phone OCR stack. Detection is PP-OCRv4 det (RapidOCR); recognisers are compared per line.

**Finding:** PP-OCR's ch v4 / latin v5 recognition dicts lack 123 / 90 of the 146 Vietnamese letters, so they cannot read Vietnamese. VietOCR does. On 6 Vietnamese legal docs (203 lines), `vgg_transformer` was near-perfect at ~1 s/line and `vgg_seq2seq` was close at ~0.1 s/line. Phone plan: PP-OCR det → VietOCR seq2seq (vi/Latin) or PP-OCR ch rec (zh).

## Run

From this folder, using the ConferenceASR venv (Python 3.12 + `rapidocr_onnxruntime vietocr torchvision onnx onnxconverter-common`):

```powershell
$env:PYTHONIOENCODING="utf-8"
..\..\ConferenceASR\.venv\Scripts\python.exe run_ocr_test.py            # photos/ -> out/report.html
..\..\ConferenceASR\.venv\Scripts\python.exe run_ocr_test.py --photos samples --out out_samples
..\..\ConferenceASR\.venv\Scripts\python.exe export_vietocr_onnx.py     # -> models/vietocr_s2s/{fp32,fp16,int8_dyn,int8_qdq}
..\..\ConferenceASR\.venv\Scripts\python.exe vietocr_onnx.py            # ONNX variants vs torch on out/results.json lines
```

Put `photo.txt` (ground truth) next to `photo.jpg` to get CER. The PP-OCRv5 latin rec used for comparison is `languages/latin/{rec.onnx,dict.txt}` from `huggingface.co/monkt/paddleocr-onnx`, placed in `models/latin_v5/`.

## VietOCR seq2seq ONNX (203 lines, vs torch)

| variant | size | =torch | CER vs torch |
|---|---|---|---|
| fp32 + LANCZOS | 89.5 MB | 202/203 | 0.00% |
| fp16 + LANCZOS | 44.7 MB | 201/203 | 0.04% |
| int8 static QDQ + LANCZOS | 27.3 MB | 165/203 | 1.3% (0.5% held-out) |
| fp32 + bilinear / area resize | 89.5 MB | 137 / 126 | 6.3% / 7.4% |

Resize to height 32 **must** be PIL LANCZOS: the model was trained on it, and bilinear/area costs more accuracy than int8 does. `vietocr_onnx.py` is the reference for the Java port (preprocess, encoder, greedy decoder loop: sos=1, eos=2, chars from index 4 = `vocab.txt`). VietOCR's vocab has no `–`, so it outputs `?` for en dashes.

# GPT-2 + LoRA PDF Fine-tune

- Fine-tune GPT-2 on your PDFs using LoRA (~0.6% params)  
- Extracts + cleans text → trains → generates matching text  
- Runs easily on Google Colab (GPU recommended)

```bash
pip install transformers datasets peft torch accelerate pdfplumber evaluate


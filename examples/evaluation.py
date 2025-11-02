import os

import torch
from transformers import RobertaTokenizer

from gpt_detector.model import ROBERTAClassifier
from gpt_detector.utils import load_checkpoint, parse_args


def evaluate(model, device, data, mask):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        mask = mask.to(device)
        data_test = data.to(device)

        y_pred = model(input_ids=data_test, attention_mask=mask)
        pred_label = torch.argmax(y_pred)

    return pred_label.item()


args = parse_args()

input_text = "this is a machine-generated text"
tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
out = tokenizer(input_text, padding="max_length", truncation=True)

data_in = out["input_ids"]
mask_in = out["attention_mask"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transfer to tensor
data_in = torch.tensor(data_in, dtype=torch.long, device=device).view(1, -1)
mask_in = torch.tensor(mask_in, dtype=torch.long, device=device).view(1, -1)

# print(mask_in.shape, data_in.shape)

# define the model
model = ROBERTAClassifier(
    n_classes=2, dropout_rate=args.dropout_rate, model_path=args.saved_model_dir
)

print("======================= Start Evaluating ==============================")
load_model_fp = os.path.join(args.output_path, "model.pkl")
model = load_checkpoint(load_model_fp, model)
pred_label = evaluate(model=model, device=device, data=data_in, mask=mask_in)

if pred_label == 0:
    print("machine generated")
else:
    print("human generated")

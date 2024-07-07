import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import gradio as gr

        
class CustomBert(nn.Module):
    def __init__(self, name_or_model_path="bert-base-uncased", num_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

model = CustomBert()
model.load_state_dict(torch.load("model_ds_netflix.pth", map_location=torch.device('cpu')))

def classifier_fn(text:str):
    labels = {0: "Movie", 1: "Series"}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, padding="max_length", max_length=250, truncation=True, return_tensors="pt")
    output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    _, pred = output.max(1)
    
    return labels[pred.item()]

demo = gr.Interface(
    fn=classifier_fn,
    inputs=["text"],
    outputs=["text"],
)


if __name__ == "__main__":
    demo.launch()
from transformers import AutoTokenizer
from train import Q_encoder, A_encoder

question = 'What is the tallest mountain in the world?'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=64)
question_emb = Q_encoder(question_tok)[0]
print(question_tok)
print('\n')
print(question_emb[:5])


answers = [
    "What is the tallest mountain in the world?",
    "The tallest mountain in the world is Mount Everest.",
    "Who is donald duck?"
]
answer_tok = []
answer_emb = []
for answer in answers:
    tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=64)
    answer_tok.append(tok['input_ids'])
    emb = A_encoder(tok)[0]
    answer_emb.append(emb)

print(answer_tok)
print(answer_emb[0][:5])
print(answer_emb[1][:5])
print(answer_emb[2][:5])


# Similarity
question_emb @ torch.stack(answer_emb).T
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
token_size = 8
embeddings_count = 32
max_iterations = 3000
evaluation_interval = 300
losses_eval_iterations = 200
learning_rate = 1e-3

# recieving input
with open('input.txt', 'r', encoding='utf-8') as open_text:
    input_text = open_text.read()

characters_list = sorted(list(set(input_text)))
chars_num = len(characters_list)

# Dictionary for encoding decoding
char_to_number = {char: index for index, char in enumerate(characters_list)}
number_to_char = {index: char for index, char in enumerate(characters_list)}


def encode_input(text):
    return [char_to_number[c] for c in text if c in char_to_number]


def decode_input(numbers):
    return ''.join([number_to_char[n] for n in numbers])


# preparing data
data = torch.tensor(encode_input(input_text), dtype=torch.long)
n = int(0.8*len(data))
training_data = data[:n]
validation_data = data[n:]


# data loading
def get_batch(split):
    split_data = training_data if split == 'train' else validation_data
    # generate batch_size number of random numbers Ex-4
    random_no = torch.randint(
        len(split_data) - token_size, (batch_size,))
    # stack the random_no.s row by row to create a matrix of Ex-[4x8]
    input_batch = torch.stack([data[i:i+token_size]
                              for i in random_no])
    target_batch = torch.stack(
        [data[i+1:i+token_size+1] for i in random_no])
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    return input_batch, target_batch


class GptModel(nn.Module):

    def __init__(self):
        super().__init__()
        # makes 32 embeddings each of 65 characters
        self.token_embedding_table = nn.Embedding(
            chars_num, embeddings_count)
        # makes 32 embeddings for each token
        self.position_embedding_table = nn.Embedding(
            token_size, embeddings_count)
        # Linear layer to convert token embeddings (32) [inputs] to vocab_size (65)(output)
        self.emb_convert = nn.Linear(embeddings_count, chars_num)

    def forward(self, index, targets=None):
        # each token in each batch contains 65 logits, which when trained, predict the possibility of next character Ex- [0.04,0.01,0.45,0.01,.....] => here 3rd character has 45% prob of being next, 1st character has 4% prob of being next
        token_embeddings = self.token_embedding_table(index)
        # token embeddings = (B,T,32)
        position_embeddings = self.position_embedding_table(
            torch.arange(token_size, device=device))
        # position_embeddings = (T, 32)
        net_token = token_embeddings + position_embeddings
        predictions = self.emb_convert(net_token)
        # predictions = (B,T, 65 = chars_num)
        if targets is None:
            loss = None
        else:
            B, T, C = predictions.shape
            # Changed the dimesnion to calculate cross entropy loss function between targets and predictions
            predictions = predictions.view(B*T, C)
            # predictions changed from Ex-(4,8,65) => (32,65)
            targets = targets.view(B*T)
            # predictions changed from Ex-(4,8) => (1, 32)
            loss = F.cross_entropy(predictions, targets)
        return predictions, loss

    def generate_new_tokens(self, input_tokens, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss_substitute = self.forward(input_tokens)
            logits = logits[:, -1, :]  # becomes (B, C), takes the last
            probs = F.softmax(logits, dim=-1)
            # (B, 1), makes the highest probability number as the only token in each batch
            idx_next = torch.multinomial(probs, num_samples=1)
            # (Ex of 1 batch)= [0.01,0.45,0.09,0.03,...], then it converts to [2] as character 2 has highest prob of 45%
            # (B, T+1), adds to make the tokens (B,T) => (B,T+1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens


# creating a model object
model = GptModel()
gpt_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    mean_loss = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(losses_eval_iterations)
        for i in range(losses_eval_iterations):
            a, b = get_batch(split)
            logits, loss = model.forward(a, b)
            losses[i] = loss.item()
        mean_loss[split] = losses.mean()
    model.train()
    return mean_loss


for i in range(max_iterations):

    # every once in a while evaluate the loss on train and val sets
    if i % evaluation_interval == 0:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_input(gpt_model.generate_new_tokens(
    context, max_new_tokens=500)[0].tolist()))

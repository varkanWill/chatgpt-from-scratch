import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
token_size = 256
embeddings_count = 384
dropout = 0.3
max_iterations = 5000
evaluation_interval = 500
losses_eval_iterations = 200
learning_rate = 3e-4
num_heads = 6
num_layer = 8

# recieving input
with open('input_conversations.txt', 'r', encoding='utf-8') as open_text:
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


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size)
                                   for _ in range(heads_count)])
        self.proj = nn.Linear(embeddings_count, embeddings_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embeddings_count, head_size, bias=False)
        self.query = nn.Linear(embeddings_count, head_size, bias=False)
        self.value = nn.Linear(embeddings_count, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(token_size, token_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        key_matrix = self.key(x)   # (B,T,head_size)
        query_matrix = self.query(x)  # (B,T,head_size)
        # compute attention scores ("affinities")
        affinity_matrix = query_matrix @ key_matrix.transpose(-2, -1) * \
            key_matrix.shape[-1]**-0.5
        affinity_matrix = affinity_matrix.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        affinity_matrix = self.dropout(affinity_matrix)
        value_matrix = self.value(x)  # (B,T,head_size)
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = affinity_matrix @ value_matrix
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.feed_fwrd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # to add an activation function (relu) that is non linear for better training and optimization
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_fwrd(x)


class Block(nn.Module):

    def __init__(self, num_embd, num_head):
        super().__init__()
        head_size = num_embd // num_head
        self.selfAttention = MultiHeadAttention(num_head, head_size)
        self.feedfrwd = FeedFoward(num_embd)
        self.layernorm1 = nn.LayerNorm(num_embd)
        self.layernorm2 = nn.LayerNorm(num_embd)

    def forward(self, x):
        x = x + self.selfAttention(self.layernorm1(x))
        x = x + self.feedfrwd(self.layernorm2(x))
        return x


class GptModel(nn.Module):

    def __init__(self):
        super().__init__()
        # makes 32 embeddings each of total number of characters
        self.token_embedding_table = nn.Embedding(
            chars_num, embeddings_count)
        # makes 32 embeddings for each token
        self.position_embedding_table = nn.Embedding(
            token_size, embeddings_count)
        # Linear layer to convert token embeddings (32) [inputs] to vocab_size
        self.emb_convert = nn.Linear(embeddings_count, chars_num)
        self.layernorm_function = nn.LayerNorm(embeddings_count)
        self.blocks = nn.Sequential(
            *[Block(embeddings_count, num_head=num_heads) for _ in range(num_layer)])

    def forward(self, index, targets=None):
        # each token in each batch contains 65 logits, which when trained, predict the possibility of next character Ex- [0.04,0.01,0.45,0.01,.....] => here 3rd character has 45% prob of being next, 1st character has 4% prob of being next
        B, T = index.shape
        token_embeddings = self.token_embedding_table(index)
        # token embeddings = (B,T, embeddings count)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device='cuda'))  # (T, embeddings count)
        # position_embeddings = (T, 32)
        net_token = token_embeddings + position_embeddings
        net_token = self.blocks(net_token)
        net_token = self.layernorm_function(net_token)
        predictions = self.emb_convert(
            net_token)
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
            # goes from -token size to -1, so that position embeddng table doesnt go out of scope
            input_tokens_cropped = input_tokens[:, -token_size:]
            logits, loss_substitute = self.forward(input_tokens_cropped)
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

    # every once in a while evaluate the loss on train and validation sets
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
    context, max_new_tokens=8000)[0].tolist()))

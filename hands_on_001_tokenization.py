with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# FIXME: implement decode function
# decode = 

print(f"chars: {chars}")
print(f"stoi: {stoi}")
print(f"vocab_size: {vocab_size}")
print(f"encode('hii there'): {encode('hii there')}")
# print(f"decode(encode('hii there')): {decode(encode('hii there'))}")

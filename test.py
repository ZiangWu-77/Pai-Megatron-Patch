import torch

# --- 1. Setup of the final, focused example ---
batch_size = 2
seq_len = 8
hidden_dim = 3

# Base embeddings tensor, format: (Seq, Batch, Hidden)
embeddings = torch.zeros(seq_len, batch_size, hidden_dim)

# --- Image Embeddings Setup ---
# This tensor concatenates all image patch embeddings, ordered by batch.
# Total patches = 3 (from batch 0) + 2 (from batch 1) = 5
image_embeds = torch.tensor([
    # --- Embeddings for Batch 0's image (3 patches) ---
    [111., 111., 111.], # Batch 0, Patch 1
    [222., 222., 222.], # Batch 0, Patch 2
    [333., 333., 333.], # Batch 0, Patch 3
    # --- Embeddings for Batch 1's image (2 patches) ---
    [444., 444., 444.], # Batch 1, Patch 1
    [555., 555., 555.]  # Batch 1, Patch 2
])

# Define the locations of the image tokens in (seq_idx, batch_idx) format.
# CRITICAL DESIGN: Batch 1's tokens appear at earlier sequence positions.
image_token_locations = [
    # Batch 0 locations (later in sequence)
    (4, 0), (5, 0), (6, 0),
    # Batch 1 locations (earlier in sequence)
    (1, 1), (2, 1)
]

print("--- Final, Focused Setup ---")
print(f"Batch size: {batch_size}, Seq length: {seq_len}")
print(f"Total image patch embeddings to insert (shape: {image_embeds.shape}):\n{image_embeds}\n")
print(f"Image token locations (seq_idx, batch_idx): {image_token_locations}\n")


# --- 2. Method 1: Advanced indexing on (Seq, Batch, Hidden) tensor ---
print("--- Method 1: Advanced Indexing on (Seq, Batch, Hidden) ---")
embeddings_m1 = embeddings.clone()

image_input_mask = torch.zeros(seq_len, batch_size, dtype=torch.bool)
for seq_idx, batch_idx in image_token_locations:
    image_input_mask[seq_idx, batch_idx] = True

# Traversal order finds Trues at (1,1), (2,1) first, then (4,0), (5,0), (6,0).
# This is the wrong order for filling from the batch-ordered `image_embeds`.
embeddings_m1[image_input_mask] = image_embeds

print("Result of Method 1 (shape: (Seq, Batch, Hidden)):")
# To make it easier to read, we permute and show each batch item's sequence
embeddings_m1_transposed = embeddings_m1.permute(1, 0, 2)
for b in range(batch_size):
    print(f"\nBatch {b} sequence from Method 1:")
    print(embeddings_m1_transposed[b, :, :])

print("\nAnalysis of Method 1:")
print(">>> FAILURE! Batch 1 (at seq pos 1, 2) received embeddings [111, 222], which belong to Batch 0.")
print(">>> Conversely, Batch 0 (at seq pos 4, 5, 6) received [333, 444, 555], mixing embeddings from both batches.")
print(">>> This is a direct result of the incorrect Seq-major traversal order.\n\n")


# --- 3. Method 2: masked_scatter on (Batch, Seq, Hidden) tensor ---
print("--- Method 2: masked_scatter on (Batch, Seq, Hidden) ---")
inputs_embeds_m2 = embeddings.permute(1, 0, 2).clone()
image_mask = torch.zeros_like(inputs_embeds_m2, dtype=torch.bool)
for seq_idx, batch_idx in image_token_locations:
    image_mask[batch_idx, seq_idx, :] = True

# The Batch-major traversal correctly processes all of Batch 0 first, then Batch 1.
inputs_embeds_m2.masked_scatter_(image_mask, image_embeds)

print("Result of Method 2 (shape: (Batch, Seq, Hidden)):")
for b in range(batch_size):
    print(f"\nBatch {b} sequence from Method 2:")
    print(inputs_embeds_m2[b, :, :])

print("\nAnalysis of Method 2:")
print(">>> CORRECT! Batch 0 correctly has [111, 222, 333] at sequence positions 4, 5, 6.")
print(">>> Batch 1 correctly has [444, 555] at sequence positions 1, 2.")
print(">>> The Batch-major processing order ensures data integrity within each sample.")

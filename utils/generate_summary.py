import numpy as np

# Knapsack algorithm implementation
def solve_knapsack(capacity, weights, values, n_items):
    dp_table = [[0] * (capacity + 1) for _ in range(n_items + 1)]
    # Build the DP table
    for i in range(1, n_items + 1):
        current_weight = weights[i - 1]
        current_value = values[i - 1]
        for w in range(1, capacity + 1):
            if current_weight <= w:
                dp_table[i][w] = max(current_value + dp_table[i - 1][w - current_weight], dp_table[i - 1][w])
            else:
                dp_table[i][w] = dp_table[i - 1][w]
    # Backtrack to find selected items
    selected_indices = []
    current_w = capacity
    for i in range(n_items, 0, -1):
        if dp_table[i][current_w] != dp_table[i - 1][current_w]:
            selected_indices.append(i - 1)
            current_w -= weights[i - 1]
    
    return selected_indices[::-1]

# Generate binary summary mask based on predicted scores and shot boundaries
def generate_summary(pred_score_list, change_points_list, n_frames_list, picks_list):
    all_summaries = []
    for segment_scores, shot_bounds, n_frames, positions in zip(pred_score_list, change_points_list, n_frames_list, picks_list):
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        # Ensure positions are integers and include the last frame index
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_start, pos_end = positions[i], positions[i + 1]
            if i == len(segment_scores):
                frame_scores[pos_start:pos_end] = 0
            else:
                frame_scores[pos_start:pos_end] = segment_scores[i]
        # Calculate shot-level scores and weights
        shot_weights = []
        shot_values = []
        for start_frame, end_frame in shot_bounds:
            length = end_frame - start_frame + 1
            shot_weights.append(length)
            mean_score = frame_scores[start_frame : end_frame + 1].mean().item()
            shot_values.append(mean_score)
        # Determine the maximum summary length (15% of total duration) and solve the knapsack problem
        last_frame_index = shot_bounds[-1][1]
        total_duration = last_frame_index + 1
        max_summary_length = int(total_duration * 0.15)
        selected_indices = solve_knapsack(max_summary_length, shot_weights, shot_values, len(shot_weights))
        # Generate binary summary mask based on selected shots
        summary_mask = np.zeros(total_duration, dtype=np.int8)
        for idx in selected_indices:
            start_f, end_f = shot_bounds[idx]
            summary_mask[start_f : end_f + 1] = 1
        all_summaries.append(summary_mask)

    return all_summaries
use crate::types::{
    label_to_index, safe_desc_order, ALPHABET_LEN, HOLDOUT_FRACTION, MIN_READY_SAMPLES_PER_LETTER,
    MIN_READY_USER_INPUTTED_PER_LETTER,
};

pub fn compute_letter_stats(
    labels: &[u8],
    acceptances: &[u8],
    out_counts: &mut [u32],
    out_ready: &mut [u8],
    out_priority: &mut [u8],
) -> i32 {
    out_counts.fill(0);
    out_ready.fill(0);
    out_priority.fill(0);

    if labels.is_empty() {
        return -1;
    }

    let mut totals = [0_u32; ALPHABET_LEN];
    let mut observed = 0_u32;
    for index in 0..labels.len() {
        let Some(label_index) = label_to_index(labels[index]) else {
            continue;
        };
        let acceptance_index = if acceptances[index] == 1 { 0 } else { 1 };
        out_counts[(label_index * 2) + acceptance_index] += 1;
        totals[label_index] += 1;
        observed += 1;
    }

    let mut most_needed: i32 = -1;
    let mut min_total = u32::MAX;
    for label_index in 0..ALPHABET_LEN {
        let user_count = out_counts[label_index * 2];
        let implicit_count = out_counts[(label_index * 2) + 1];
        if user_count as usize >= MIN_READY_USER_INPUTTED_PER_LETTER
            && (user_count + implicit_count) as usize >= MIN_READY_SAMPLES_PER_LETTER
        {
            out_ready[label_index] = 1;
        }

        let total = totals[label_index];
        if observed > 0 && total < min_total {
            min_total = total;
            most_needed = label_index as i32;
        }
    }

    if observed == 0 {
        return -1;
    }

    let average = observed as f64 / ALPHABET_LEN as f64;
    for label_index in 0..ALPHABET_LEN {
        if (totals[label_index] as f64) < average * 0.85 {
            out_priority[label_index] = 1;
        }
    }

    most_needed
}

fn select_holdout_count(count: usize) -> usize {
    if count < MIN_READY_SAMPLES_PER_LETTER {
        0
    } else {
        ((count as f64 * HOLDOUT_FRACTION).floor() as usize).max(1)
    }
}

pub fn build_balanced_dataset(
    labels: &[u8],
    acceptances: &[u8],
    created_at: &[f64],
    out_training_mask: &mut [u8],
    out_holdout_mask: &mut [u8],
    out_ready: &mut [u8],
) -> u32 {
    out_training_mask.fill(0);
    out_holdout_mask.fill(0);
    out_ready.fill(0);

    if labels.is_empty() {
        return 0;
    }

    let mut sorted_indices: Vec<usize> = (0..labels.len()).collect();
    sorted_indices.sort_by(|left, right| safe_desc_order(created_at[*left], created_at[*right]));

    let mut user_buckets: [Vec<usize>; ALPHABET_LEN] = std::array::from_fn(|_| Vec::new());
    let mut implicit_buckets: [Vec<usize>; ALPHABET_LEN] = std::array::from_fn(|_| Vec::new());

    for index in sorted_indices {
        let Some(label_index) = label_to_index(labels[index]) else {
            continue;
        };
        if acceptances[index] == 1 {
            user_buckets[label_index].push(index);
        } else {
            implicit_buckets[label_index].push(index);
        }
    }

    let ready_letters: Vec<usize> = (0..ALPHABET_LEN)
        .filter(|label_index| {
            user_buckets[*label_index].len() >= MIN_READY_USER_INPUTTED_PER_LETTER
                && (user_buckets[*label_index].len() + implicit_buckets[*label_index].len())
                    >= MIN_READY_SAMPLES_PER_LETTER
        })
        .collect();

    if ready_letters.len() < 2 {
        return 0;
    }

    for label_index in &ready_letters {
        out_ready[*label_index] = 1;
    }

    let per_letter_target = ready_letters
        .iter()
        .map(|label_index| user_buckets[*label_index].len() + implicit_buckets[*label_index].len())
        .min()
        .unwrap_or(0);
    if per_letter_target == 0 {
        return 0;
    }

    let target_user = ((per_letter_target as f64) * 0.2).round() as usize;
    let target_user = target_user.max(1);

    for label_index in ready_letters {
        let user_bucket = &user_buckets[label_index];
        let implicit_bucket = &implicit_buckets[label_index];
        let user_holdout_count = select_holdout_count(user_bucket.len());
        let implicit_holdout_count = select_holdout_count(implicit_bucket.len());

        for index in user_bucket
            .iter()
            .skip(user_bucket.len().saturating_sub(user_holdout_count))
        {
            out_holdout_mask[*index] = 1;
        }
        for index in implicit_bucket
            .iter()
            .skip(implicit_bucket.len().saturating_sub(implicit_holdout_count))
        {
            out_holdout_mask[*index] = 1;
        }

        let user_pool = &user_bucket[..user_bucket.len().saturating_sub(user_holdout_count)];
        let implicit_pool =
            &implicit_bucket[..implicit_bucket.len().saturating_sub(implicit_holdout_count)];

        let chosen_user_len = user_pool.len().min(target_user);
        let implicit_target = per_letter_target.saturating_sub(chosen_user_len);
        let chosen_implicit_len = implicit_pool.len().min(implicit_target);

        let mut selected = Vec::with_capacity(per_letter_target);
        selected.extend_from_slice(&user_pool[..chosen_user_len]);
        selected.extend_from_slice(&implicit_pool[..chosen_implicit_len]);

        if selected.len() < per_letter_target {
            selected.extend_from_slice(&implicit_pool[chosen_implicit_len..]);
        }
        if selected.len() < per_letter_target {
            selected.extend_from_slice(&user_pool[chosen_user_len..]);
        }

        for index in selected.into_iter().take(per_letter_target) {
            out_training_mask[index] = 1;
        }
    }

    per_letter_target as u32
}

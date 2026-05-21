use crate::types::{label_to_index, ALPHABET_LEN};

const MIN_READY_SAMPLES_PER_LETTER: u32 = 5;
const MIN_READY_USER_INPUTTED_PER_LETTER: u32 = 1;
const HOLDOUT_FRACTION: f64 = 0.2;

#[derive(Clone, Copy)]
struct Counts {
    implicit: u32,
    user_inputted: u32,
}

impl Counts {
    fn total(self) -> u32 {
        self.implicit + self.user_inputted
    }

    fn ready(self) -> bool {
        self.user_inputted >= MIN_READY_USER_INPUTTED_PER_LETTER
            && self.total() >= MIN_READY_SAMPLES_PER_LETTER
    }
}

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

    for (label, acceptance) in labels.iter().zip(acceptances.iter()) {
        let Some(index) = label_to_index(*label) else {
            continue;
        };
        let offset = index * 2;
        if *acceptance == 1 {
            out_counts[offset] += 1;
        } else {
            out_counts[offset + 1] += 1;
        }
    }

    let mut ready_count = 0_i32;
    for index in 0..ALPHABET_LEN {
        let counts = Counts {
            user_inputted: out_counts[index * 2],
            implicit: out_counts[index * 2 + 1],
        };
        if counts.ready() {
            out_ready[index] = 1;
            ready_count += 1;
        }
    }

    let total_samples = (0..ALPHABET_LEN)
        .map(|index| out_counts[index * 2] + out_counts[index * 2 + 1])
        .sum::<u32>();
    if total_samples > 0 {
        let average = total_samples as f64 / ALPHABET_LEN as f64;
        for index in 0..ALPHABET_LEN {
            let total = out_counts[index * 2] + out_counts[index * 2 + 1];
            if (total as f64) < average {
                out_priority[index] = 1;
            }
        }
    }

    ready_count
}

fn select_holdout_count(count: u32) -> u32 {
    if count < MIN_READY_SAMPLES_PER_LETTER {
        0
    } else {
        ((count as f64 * HOLDOUT_FRACTION).floor() as u32).max(1)
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

    let mut buckets = vec![(Vec::<usize>::new(), Vec::<usize>::new()); ALPHABET_LEN];
    for index in 0..labels.len() {
        let Some(label_index) = label_to_index(labels[index]) else {
            continue;
        };
        if acceptances[index] == 1 {
            buckets[label_index].0.push(index);
        } else {
            buckets[label_index].1.push(index);
        }
    }

    for (user_inputted, implicit) in buckets.iter_mut() {
        user_inputted.sort_by(|left, right| {
            created_at[*right]
                .partial_cmp(&created_at[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        implicit.sort_by(|left, right| {
            created_at[*right]
                .partial_cmp(&created_at[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let ready: Vec<usize> = buckets
        .iter()
        .enumerate()
        .filter_map(|(index, (user_inputted, implicit))| {
            let counts = Counts {
                user_inputted: user_inputted.len() as u32,
                implicit: implicit.len() as u32,
            };
            counts.ready().then_some(index)
        })
        .collect();

    if ready.len() < 2 {
        return 0;
    }

    for index in &ready {
        out_ready[*index] = 1;
    }

    let per_letter_target = ready
        .iter()
        .map(|index| buckets[*index].0.len() + buckets[*index].1.len())
        .min()
        .unwrap_or(0) as u32;
    let target_user_inputted = ((per_letter_target as f64) * 0.2).round().max(1.0) as usize;

    for label_index in ready {
        let (user_inputted, implicit) = &buckets[label_index];
        let user_holdout_count = select_holdout_count(user_inputted.len() as u32) as usize;
        let implicit_holdout_count = select_holdout_count(implicit.len() as u32) as usize;

        for sample_index in user_inputted.iter().rev().take(user_holdout_count) {
            out_holdout_mask[*sample_index] = 1;
        }
        for sample_index in implicit.iter().rev().take(implicit_holdout_count) {
            out_holdout_mask[*sample_index] = 1;
        }

        let user_pool_end = user_inputted.len().saturating_sub(user_holdout_count);
        let implicit_pool_end = implicit.len().saturating_sub(implicit_holdout_count);
        let user_pool = &user_inputted[..user_pool_end];
        let implicit_pool = &implicit[..implicit_pool_end];

        let mut selected = Vec::new();
        selected.extend(user_pool.iter().take(target_user_inputted).copied());
        let implicit_target = (per_letter_target as usize).saturating_sub(selected.len());
        selected.extend(implicit_pool.iter().take(implicit_target).copied());

        if selected.len() < per_letter_target as usize {
            for sample_index in implicit_pool.iter().chain(user_pool.iter()) {
                if selected.contains(sample_index) {
                    continue;
                }
                selected.push(*sample_index);
                if selected.len() >= per_letter_target as usize {
                    break;
                }
            }
        }

        for sample_index in selected.into_iter().take(per_letter_target as usize) {
            out_training_mask[sample_index] = 1;
        }
    }

    per_letter_target
}

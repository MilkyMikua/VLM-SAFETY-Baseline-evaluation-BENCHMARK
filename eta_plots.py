import argparse
import json
import os
import matplotlib.pyplot as plt

def loadj(p):
    with open(p, 'r') as f:
        return json.load(f)

def plot_recall(m, title, out_path):
    labels = ['safe_pre','safe_post','unsafe_pre','unsafe_post']
    vals = [m['utility_recall@k_pre_S_V'],m['utility_recall@k_post_S_V'],m['harmful_recall@k_pre_U_V'],m['harmful_recall@k_post_U_V']]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, vals, color=['#4caf50','#81c784','#e53935','#ef9a9a'])
    ax.set_ylim(0,1)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_shift(m, title, out_path):
    labels = ['decline_to_unsafe','increase_to_neutral']
    vals = [m['text_semantic_shift_decline_to_unsafe'],m['text_semantic_shift_increase_to_neutral']]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, vals, color=['#fb8c00','#42a5f5'])
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v+(0.02 if v>=0 else -0.06), f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_clip(m, title, out_path):
    labels = ['clipscore_safe_pre','clipscore_safe_post']
    vals = [m['clipscore_safe_pre'],m['clipscore_safe_post']]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, vals, color=['#8e24aa','#ce93d8'])
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--dev_json', type=str, required=True)
    ap.add_argument('--test_json', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    m_train = loadj(args.train_json)
    m_dev = loadj(args.dev_json)
    m_test = loadj(args.test_json)
    plot_recall(m_train,'ETA Recall@K (Train)',os.path.join(args.out_dir,'eta_recall_train.png'))
    plot_recall(m_dev,'ETA Recall@K (Dev)',os.path.join(args.out_dir,'eta_recall_dev.png'))
    plot_recall(m_test,'ETA Recall@K (Test)',os.path.join(args.out_dir,'eta_recall_test.png'))
    plot_shift(m_train,'ETA Semantic Shift (Train)',os.path.join(args.out_dir,'eta_shift_train.png'))
    plot_shift(m_dev,'ETA Semantic Shift (Dev)',os.path.join(args.out_dir,'eta_shift_dev.png'))
    plot_shift(m_test,'ETA Semantic Shift (Test)',os.path.join(args.out_dir,'eta_shift_test.png'))
    plot_clip(m_train,'ETA CLIPScore (Train)',os.path.join(args.out_dir,'eta_clip_train.png'))
    plot_clip(m_dev,'ETA CLIPScore (Dev)',os.path.join(args.out_dir,'eta_clip_dev.png'))
    plot_clip(m_test,'ETA CLIPScore (Test)',os.path.join(args.out_dir,'eta_clip_test.png'))

if __name__ == '__main__':
    main()


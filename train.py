import torch
from torch import nn, optim

from data import get_dataloaders
from metric import calc_nd
from models import DomainDiscriminator, SequenceGenerator, SharedAttention
from utils import make_true_dom


def train(
    feat_dim,
    pred_len,
    hidden_dim,
    kernel_size,
    syn_type,
    syn_param,
    batch_size,
    tradeoff,
    num_epoch,
    lr,
):
    print(f"Training with {syn_type}-{syn_param} data")
    print(
        f"  feat_dim: {feat_dim}, pred_len: {pred_len}, hidden_dim: {hidden_dim}, kernel_size: {kernel_size}"
    )
    print(f"  batch_size: {batch_size}, num_epoch: {num_epoch}, lr: {lr}")
    print(f"  tradeoff: {tradeoff}\n")

    # configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    src_trainloader, tgt_trainloader, tgt_validloader = get_dataloaders(
        syn_type, syn_param, feat_dim, pred_len, batch_size
    )

    # models
    shr_attention = SharedAttention(feat_dim, hidden_dim, kernel_size)

    src_generator, tgt_generator = (
        SequenceGenerator(feat_dim, pred_len, shr_attention, hidden_dim, kernel_size),
        SequenceGenerator(feat_dim, pred_len, shr_attention, hidden_dim, kernel_size),
    )
    dom_discriminator = DomainDiscriminator(feat_dim, hidden_dim)

    # optimizers
    mse, bce = nn.MSELoss(), nn.BCELoss()
    att_optim = optim.Adam(shr_attention.parameters(), lr=lr)
    gen_optim = optim.Adam(
        list(src_generator.enc.parameters())
        + list(src_generator.dec.parameters())
        + list(tgt_generator.enc.parameters())
        + list(tgt_generator.dec.parameters()),
        lr=lr,
    )
    dis_optim = optim.Adam(dom_discriminator.parameters(), lr=lr)

    # training
    for model in [shr_attention, src_generator, tgt_generator, dom_discriminator]:
        model.train()
        model.to(device)
    for epoch in range(num_epoch):
        train_seq_losses, train_dom_losses, train_tot_losses = [], [], []
        for (src_data, src_true), (tgt_data, tgt_true) in zip(
            src_trainloader, tgt_trainloader
        ):
            src_data, src_true, tgt_data, tgt_true = (
                src_data.to(device),
                src_true.to(device),
                tgt_data.to(device),
                tgt_true.to(device),
            )

            gen_optim.zero_grad()
            dis_optim.zero_grad()
            att_optim.zero_grad()

            # reconstruction & prediction
            src_pred, (src_query, src_key) = src_generator(src_data)
            tgt_pred, (tgt_query, tgt_key) = tgt_generator(tgt_data)

            # domain classification
            src_dom_q, src_dom_k = dom_discriminator(src_query, src_key)
            tgt_dom_q, tgt_dom_k = dom_discriminator(tgt_query, tgt_key)
            src_dom, tgt_dom = make_true_dom(src_dom_q, tgt_dom_q)

            # loss calculation
            seq_loss = (
                mse(src_data, src_pred[..., :-pred_len]).mean()
                + mse(src_true, src_pred[..., -pred_len:]).mean()
                + mse(tgt_data, tgt_pred[..., :-pred_len]).mean()
                + mse(tgt_true, tgt_pred[..., -pred_len:]).mean()
            )
            dom_loss = -(
                (bce(src_dom_q, src_dom) + bce(src_dom_k, src_dom)).mean()
                + (bce(tgt_dom_q, tgt_dom) + bce(tgt_dom_k, tgt_dom)).mean()
            )
            tot_loss = seq_loss - tradeoff * dom_loss
            train_seq_losses.append(seq_loss.item())
            train_dom_losses.append(dom_loss.item())
            train_tot_losses.append(tot_loss.item())

            # backpropagation
            tot_loss.backward()
            gen_optim.step()
            dis_optim.step()
            att_optim.step()

        metrics = []
        for tgt_data, tgt_true in tgt_validloader:
            tgt_data, tgt_true = tgt_data.to(device), tgt_true.to(device)
            tgt_pred, (tgt_query, tgt_key) = tgt_generator(tgt_data)
            norm_devn = calc_nd(tgt_true, tgt_pred[..., -pred_len:])
            metrics.append(norm_devn.item())

        print(f"Epoch {epoch + 1:04} /{num_epoch} {'=' * 30}")
        print(f"Metric: valid {sum(metrics) / len(metrics):.8f}")
        print(
            "Loss:"
            f" total {sum(train_tot_losses) / len(train_tot_losses):.4f}"
            f" seq {sum(train_seq_losses) / len(train_seq_losses):.4f}"
            f" dom {sum(train_dom_losses) / len(train_dom_losses):.4f}"
        )


def main():
    import argparse
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dim", type=int, default=1)
    parser.add_argument("--pred_len", type=int, default=18)
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=(64, 64))
    parser.add_argument("--kernel_size", type=int, default=(3, 5))
    parser.add_argument("--syn_type", type=str, default="coldstart")
    parser.add_argument("--syn_param", type=int, default=36)
    parser.add_argument("--batch_size", type=int, default=int(1e3))
    parser.add_argument("--num_epoch", type=int, default=int(1e3))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tradeoff", type=float, default=1.0)
    args = parser.parse_args()

    if len(args.hidden_dim) == 1:
        args.hidden_dim = args.hidden_dim[0]
    else:
        args.hidden_dim = tuple(args.hidden_dim)

    train(**vars(args))


if __name__ == "__main__":
    main()

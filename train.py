import torch
from torch import nn, optim
from tqdm import tqdm

from data import get_dataloaders
from metric import calc_nd
from models import DomainDiscriminator, SequenceGenerator, SharedAttention


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
        f"feat_dim: {feat_dim}, pred_len: {pred_len}, hidden_dim: {hidden_dim}, kernel_size: {kernel_size}"
    )
    print(f"batch_size: {batch_size}, num_epoch: {num_epoch}, lr: {lr}")
    print(f"tradeoff: {tradeoff}")

    # configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    src_trainloader, tgt_trainloader, tgt_validloader = get_dataloaders(
        syn_type=syn_type, syn_param=syn_param, pred_len=pred_len, batch_size=batch_size
    )

    # models
    shared_attn = SharedAttention(
        feat_dim=feat_dim, hidden_dim=hidden_dim, kernel_size=kernel_size
    )

    src_generator, tgt_generator = (
        SequenceGenerator(
            feat_dim=feat_dim,
            pred_len=pred_len,
            attn_module=shared_attn,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        ),
        SequenceGenerator(
            feat_dim=feat_dim,
            pred_len=pred_len,
            attn_module=shared_attn,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        ),
    )
    discriminator = DomainDiscriminator(feat_dim=feat_dim, hidden_dim=hidden_dim)

    # optimizers
    mse, bce = nn.MSELoss(), nn.BCELoss()
    att_optim = optim.Adam(shared_attn.parameters(), lr=lr)
    gen_optim = optim.Adam(
        list(src_generator.parameters()) + list(tgt_generator.parameters()),
        lr=lr,
    )
    dis_optim = optim.Adam(discriminator.parameters(), lr=lr)

    # training
    for model in [shared_attn, src_generator, tgt_generator, discriminator]:
        model.train()
        model.to(device)
    for epoch in range(num_epoch):
        train_seq_losses, train_dom_losses, train_tot_losses, train_metrics = (
            [],
            [],
            [],
            [],
        )
        for (src_data, src_true), (tgt_data, tgt_true) in tqdm(
            zip(src_trainloader, tgt_trainloader)
        ):
            src_data, src_true, tgt_data, tgt_true = (
                src_data.to(device),
                src_true.to(device),
                tgt_data.to(device),
                tgt_true.to(device),
            )

            att_optim.zero_grad()
            gen_optim.zero_grad()
            dis_optim.zero_grad()

            # reconstruction & prediction
            src_pred = src_generator(src_data)
            tgt_pred = tgt_generator(tgt_data)

            # domain classification
            src_cls_q, src_cls_k = discriminator(src_pred)
            tgt_cls_q, tgt_cls_k = discriminator(tgt_pred)

            # loss calculation
            ## estimation error
            src_seq_loss = (
                mse(src_data, src_pred[:-pred_len]).mean()
                + mse(src_true, src_pred[-pred_len:]).mean()
            )
            tgt_seq_loss = (
                mse(tgt_data, tgt_pred[:-pred_len]).mean()
                + mse(tgt_true, tgt_pred[-pred_len:]).mean()
            )
            seq_loss = src_seq_loss + tgt_seq_loss
            ## domain classification error
            src_dom_loss = -(
                bce(src_cls_q, torch.zeros_like(src_cls_q))
                + bce(src_cls_k, torch.zeros_like(src_cls_k))
            ).mean()
            tgt_dom_loss = -(
                bce(tgt_cls_q, torch.ones_like(tgt_cls_q))
                + bce(tgt_cls_k, torch.ones_like(tgt_cls_k))
            ).mean()
            dom_loss = src_dom_loss + tgt_dom_loss
            tot_loss = seq_loss - tradeoff * dom_loss
            train_seq_losses.append(seq_loss.item())
            train_dom_losses.append(dom_loss.item())
            train_tot_losses.append(tot_loss.item())

            # backpropagation
            tot_loss.backward()
            gen_optim.step()
            dis_optim.step()
            att_optim.step()

            # evaluation
            norm_devn = calc_nd(tgt_true, tgt_pred)
            train_metrics.append(norm_devn.item())

        valid_metrics = []
        for tgt_data, tgt_true in tqdm(tgt_validloader):
            tgt_data, tgt_true = tgt_data.to(device), tgt_true.to(device)
            tgt_pred = tgt_generator(tgt_data)
            norm_devn = calc_nd(tgt_true, tgt_pred)
            valid_metrics.append(norm_devn.item())

        print(f"Epoch {epoch} /{num_epoch} ====================")
        print(f"Valid metric: {sum(valid_metrics) / len(valid_metrics)}")
        print(f"Train metric: {sum(train_metrics) / len(train_metrics)}")
        print(f"Train loss: {sum(train_tot_losses) / len(train_tot_losses)}")
        print(f"  Seq loss: {sum(train_seq_losses) / len(train_seq_losses)}")
        print(f"  Dom loss: {sum(train_dom_losses) / len(train_dom_losses)}")


def main(args):
    train(**vars(args))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dim", type=int, default=1)
    parser.add_argument("--pred_len", type=int, default=18)
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=(64, 64))
    parser.add_argument("--kernel_size", type=int, default=(3, 5))
    parser.add_argument("--syn_type", type=str, default="coldstart")
    parser.add_argument("--syn_param", type=int, default=36)
    parser.add_argument("--batch_size", type=int, default=int(2e3))
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tradeoff", type=float, default=1.0)
    args = parser.parse_args()

    if len(args.hidden_dim) == 1:
        args.hidden_dim = args.hidden_dim[0]
    else:
        args.hidden_dim = tuple(args.hidden_dim)

    main(args)

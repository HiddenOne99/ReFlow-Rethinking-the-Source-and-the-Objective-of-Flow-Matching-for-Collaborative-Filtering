import os
import sys
import math
import time
import random
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn


CONFIG = SimpleNamespace(
    train_path='path/to/train',
    valid_path='path/to/valid',
    test_path='path/to/test',

    dims_mlp=[600],
    time_embedding_size=10,
    n_steps=50,
    s_steps=2,
    mlp_dropout=0.0,
    act_func='tanh',
    norm=False,

    rank_lambda=0.01,           # weight lambda of the ranking term
    rank_num_neg=50,           # negatives per positive
    rank_tau=0.2,              # ranking temperature

    alpha=0.5,                 # weight of the collaborative profile in the echo source
    kappa=5,                  # partners kept per item in the partner graph

    batch_size=4096,           # users per training batch
    eval_batch_size=2048,
    max_epochs=200,
    patience=10,
    lr=1e-3,
    weight_decay=0.0,

    seed=2027,
    eval_every=1,
    gpu=0,
    deterministic=True,
    save_path='',
    log_path='reflow_log.txt',
)

warnings.filterwarnings('ignore')

class FileLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        with open(filename, "w"):
            pass
        self.log = open(filename, "a", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self, *a, **kw):
        self.terminal.flush()
        self.log.flush()


def resolve_device(gpu):
    if gpu < 0 or not torch.cuda.is_available():
        print("[device] CPU")
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if gpu >= n:
        print(f"[device] gpu={gpu} but only {n} visible; falling back to 0")
        gpu = 0
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    return device


def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_interactions(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, engine='c',
                     usecols=['user_id', 'item_id'],
                     dtype={'user_id': np.int64, 'item_id': np.int64})
    return df.rename(columns={'user_id': 'user', 'item_id': 'item'})


def build_mappings_from_train(train_df):
    users = sorted(train_df['user'].unique())
    items = sorted(train_df['item'].unique())
    return ({u: i for i, u in enumerate(users)},
            {x: i for i, x in enumerate(items)})


def df_to_csr(df, user_map, item_map, num_users, num_items):
    df = df[df['user'].isin(user_map) & df['item'].isin(item_map)]
    r = df['user'].map(user_map).values
    c = df['item'].map(item_map).values
    d = np.ones(len(df), dtype=np.float32)
    m = sp.coo_matrix((d, (r, c)), shape=(num_users, num_items)).tocsr()
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def fmt_metrics(m):
    return "  ".join(f"{name}@{K}={m.get(f'{key}@{K}', 0):.4f}"
                     for K in (10, 20) for name, key in (('R', 'recall'), ('N', 'ndcg')))


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def xavier_normal_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)


class FlowModel(nn.Module):
    def __init__(self, dims, time_emb_size, act_func="tanh", norm=False, dropout=0.1):
        super().__init__()
        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        dims = list(dims)
        dims[0] = dims[0] + self.time_emb_dim
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[act_func]
        modules = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(d_in, d_out))
            modules.append(act())
        modules.pop()
        self.encoder = nn.Sequential(*modules[:-2]) if len(modules) > 2 else nn.Identity()
        self.head = nn.Sequential(*modules[-2:])
        self.apply(xavier_normal_init)

    def forward(self, x, t):
        emb = self.emb_layer(timestep_embedding_pi(t, self.time_emb_dim).to(x.device))
        if self.norm:
            x = nn.functional.normalize(x)
        return self.head(self.encoder(torch.cat([x, emb], dim=-1)))


def build_partner_graph(train_csr, topk):
    X = (train_csr > 0).astype(np.float32).tocsr()
    I = X.shape[1]
    n = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    G = (X.T @ X).tocsr()
    del X
    G.data = G.data.astype(np.float32, copy=False)
    rows = np.repeat(np.arange(I, dtype=np.int32), np.diff(G.indptr))
    G.data[G.indices == rows] = 0.0
    d = (1.0 / np.sqrt(np.maximum(n, 1.0))).astype(np.float32)
    G.data *= d[rows]
    G.data *= d[G.indices]
    del rows
    G.eliminate_zeros()
    if 0 < topk < I - 1:
        keep = np.zeros(G.nnz, dtype=bool)
        ip, val = G.indptr, G.data
        for i in range(I):
            a, b = ip[i], ip[i + 1]
            if b - a <= topk:
                keep[a:b] = True
            else:
                v = val[a:b]
                keep[a:b] = v >= np.partition(v, b - a - topk)[b - a - topk]
        G.data[~keep] = 0.0
        del keep
        G.eliminate_zeros()
    deg = np.asarray(G.sum(axis=1)).ravel().astype(np.float32)
    P_rw = (sp.diags((1.0 / np.maximum(deg, 1e-12)).astype(np.float32)) @ G).tocsr()
    return P_rw


class ReFlow(nn.Module):
    def __init__(self, num_users, num_items, train_csr, cfg, device):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_steps = cfg.n_steps
        self.s_steps = cfg.s_steps
        self.device = device
        self.register_buffer("time_steps", torch.linspace(0, 1, self.n_steps + 1))

        self.flow_model = FlowModel(
            dims=[num_items] + list(cfg.dims_mlp) + [num_items],
            time_emb_size=cfg.time_embedding_size,
            act_func=cfg.act_func,
            norm=cfg.norm,
            dropout=cfg.mlp_dropout,
        )
        item_users = np.asarray((train_csr > 0).sum(axis=0)).ravel().astype(np.float32)
        self.register_buffer("prior_probs", torch.from_numpy(item_users / max(num_users, 1)))

        self.rank_lambda = float(cfg.rank_lambda)
        self.rank_num_neg = int(cfg.rank_num_neg)
        self.rank_tau = float(cfg.rank_tau)
        if self.rank_tau <= 0 or self.rank_num_neg < 1:
            raise ValueError("rank_tau must be > 0 and rank_num_neg >= 1")

        self.alpha = float(cfg.alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.partner_PT = None
        if self.alpha > 0:
            PT = build_partner_graph(train_csr, int(cfg.kappa)).T.tocsr()
            self.partner_PT = torch.sparse_csr_tensor(
                torch.from_numpy(PT.indptr.astype(np.int64)), torch.from_numpy(PT.indices.astype(np.int64)),
                torch.from_numpy(PT.data.astype(np.float32)), size=PT.shape).to(device)
            del PT

    @property
    def out_layer(self):
        return [m for m in self.flow_model.head.modules() if isinstance(m, nn.Linear)][-1]

    @property
    def out_bias(self):
        return self.out_layer.bias

    def forward(self, x, t):
        return self.flow_model(x, t)

    def ranking_loss(self, s, x1):
        B, N = s.shape
        pos = x1 > 0
        u, p = pos.nonzero(as_tuple=True)
        if u.numel() == 0:
            return s.sum() * 0.0
        P, k = u.numel(), self.rank_num_neg
        j = torch.randint(0, N, (P * k,), device=s.device).view(P, k)
        uu = u.unsqueeze(1).expand(P, k)
        bad = pos[uu, j]
        for _ in range(20):
            nb = int(bad.sum())
            if nb == 0:
                break
            j[bad] = torch.randint(0, N, (nb,), device=s.device)
            bad = pos[uu, j]
        valid = (~bad).float()
        diff = (s[uu, j] - s[u, p].unsqueeze(1)) / self.rank_tau
        per_pos = (torch.nn.functional.softplus(diff) * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return per_pos.mean()

    def source_probs(self, x1):
        f = self.prior_probs.unsqueeze(0)
        if self.partner_PT is None:
            return f.expand(x1.size(0), -1)
        s = torch.sparse.mm(self.partner_PT, x1.t().contiguous()).t()
        s = s * (1.0 - x1)
        F = f.sum()
        q = s * (F / s.sum(1, keepdim=True).clamp(min=1e-12))
        return ((1.0 - self.alpha) * f + self.alpha * q).clamp(0.0, 1.0)

    def flow_loss(self, x1):
        B = x1.size(0)
        steps = torch.randint(0, self.n_steps, (B,), device=x1.device)
        t = self.time_steps[steps].unsqueeze(1)
        x0 = torch.bernoulli(self.source_probs(x1))
        mask = torch.rand_like(x1) <= t
        xt = torch.where(mask, x1, x0)
        out = self.forward(xt, t.squeeze(-1))
        loss = ((x1 - out) ** 2).mean()
        if self.rank_lambda > 0:
            s = out - self.out_bias.unsqueeze(0)
            loss = loss + self.rank_lambda * self.ranking_loss(s, x1)
        return loss

    @torch.no_grad()
    def predict_scores(self, X_bar):
        Xt = X_bar
        X1_hat = None
        for i_t in range(self.n_steps - self.s_steps, self.n_steps):
            t = self.time_steps[i_t].repeat(Xt.shape[0], 1)
            X1_hat = self.forward(Xt, t.squeeze(-1))
            if i_t == self.n_steps - 1:
                break
            t_next = self.time_steps[i_t + 1].repeat(Xt.shape[0], 1)
            v = (X1_hat - Xt) / (1 - t)
            Xt_pos = Xt + v * (t_next - t)
            Xt = torch.stack([1 - Xt_pos, Xt_pos], dim=-1).argmax(dim=-1)
            Xt = torch.logical_or(X_bar.to(torch.bool), Xt.to(torch.bool)).float()
        return X1_hat


class StreamMetrics:
    def __init__(self, eval_csr, exclude_csrs, K_list, device):
        self.eval_csr, self.exclude_csrs, self.K_list = eval_csr, exclude_csrs, tuple(K_list)
        self.maxK = max(self.K_list)
        self.disc = 1.0 / torch.log2(torch.arange(2, self.maxK + 2, device=device, dtype=torch.float64))
        self.cdisc = torch.cumsum(self.disc, 0)
        self.rec = {K: 0.0 for K in self.K_list}
        self.ndcg = {K: 0.0 for K in self.K_list}
        self.n = 0

    @torch.no_grad()
    def add(self, batch, scores):
        sc = scores.float().clone()
        for csr in self.exclude_csrs:
            sub = csr[batch].tocoo()
            if sub.nnz:
                sc[torch.as_tensor(sub.row, dtype=torch.long, device=sc.device),
                   torch.as_tensor(sub.col, dtype=torch.long, device=sc.device)] = -float('inf')
        top = sc.topk(self.maxK, dim=1).indices
        T = torch.from_numpy(self.eval_csr[batch].toarray()).to(sc.device) > 0
        hit = T.gather(1, top).double()
        npos = T.sum(1).double().clamp(min=1.0)
        for K in self.K_list:
            h = hit[:, :K]
            self.rec[K] += float((h.sum(1) / npos).sum())
            idcg = self.cdisc[(npos.clamp(max=K) - 1).long()]
            self.ndcg[K] += float(((h * self.disc[:K]).sum(1) / idcg).sum())
        self.n += len(batch)

    def result(self):
        out = {}
        for K in self.K_list:
            out[f'recall@{K}'] = self.rec[K] / max(self.n, 1)
            out[f'ndcg@{K}'] = self.ndcg[K] / max(self.n, 1)
        return out


@torch.no_grad()
def evaluate(model, train_csr, eval_csr, exclude_csrs, device, eval_batch_size, K_list=(10, 20)):
    users = sorted(set(eval_csr.tocoo().row.tolist()))
    if not users:
        return {}
    model.eval()
    acc = StreamMetrics(eval_csr, exclude_csrs, K_list, device)
    for s in range(0, len(users), eval_batch_size):
        batch = users[s:s + eval_batch_size]
        X_bar = torch.from_numpy(train_csr[batch].toarray()).to(device)
        acc.add(batch, model.predict_scores(X_bar))
    return acc.result()


def train_reflow(args, model, train_csr, valid_csr, test_csr, num_users, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_valid, best_epoch, best_state, patience_counter = -1.0, 0, None, 0
    epoch_times = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(num_users)
        num_batches = (num_users + args.batch_size - 1) // args.batch_size
        ep_loss = 0.0
        for b in range(num_batches):
            uidx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x1 = torch.from_numpy(train_csr[uidx].toarray()).to(device)
            loss = model.flow_loss(x1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().item())
        ep_loss /= max(num_batches, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_times.append(time.time() - t0)
        print(f"=== Epoch {epoch}/{args.max_epochs}  loss={ep_loss:.5f}  ({epoch_times[-1]:.1f}s) ===")

        if epoch % args.eval_every == 0:
            m = evaluate(model, train_csr, valid_csr, [train_csr], device, args.eval_batch_size)
            print(f"  valid      {fmt_metrics(m)}")
            if m.get('ndcg@10', 0) > best_valid:
                best_valid, best_epoch, patience_counter = m.get('ndcg@10', 0), epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  [early stop at epoch {epoch}; best epoch was {best_epoch}]")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if args.save_path:
            d = os.path.dirname(args.save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(best_state, args.save_path)
            print(f"[save] selected model (epoch {best_epoch}) -> {args.save_path}")

    vm = evaluate(model, train_csr, valid_csr, [train_csr], device, args.eval_batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t_inf = time.time()
    tm = evaluate(model, train_csr, test_csr, [train_csr, valid_csr], device, args.eval_batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t_inf = time.time() - t_inf
    print(f"\n=== RESULT (best epoch {best_epoch}, selected on valid N@10) ===")
    print(f"  valid      {fmt_metrics(vm)}")
    print(f"  test       {fmt_metrics(tm)}")
    print(f"\n[summary] model=ReFlow  best_epoch={best_epoch} | "
          f"rank_lambda={args.rank_lambda} K={args.rank_num_neg} tau={args.rank_tau} | "
          f"alpha={args.alpha} kappa={args.kappa} | s_steps={args.s_steps}")
    print(f"[summary]   {fmt_metrics(tm)}")
    mem = (f"  | peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
           if device.type == 'cuda' else "")
    print(f"[time] train {np.mean(epoch_times):.2f}s/epoch (mean over {len(epoch_times)} epochs)"
          f"  | test inference {t_inf:.2f}s for {test_csr.shape[0]} users (full pass, S={args.s_steps}){mem}")
    return best_epoch, tm


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('yes', 'true', 't', 'y', '1'):
        return True
    if s in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{v}'")


def build_parser():
    p = argparse.ArgumentParser(prog='reflow.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = p.add_argument_group('data')
    d.add_argument('--train_path', type=str, default=None)
    d.add_argument('--valid_path', type=str, default=None)
    d.add_argument('--test_path', type=str, default=None)

    a = p.add_argument_group('architecture')
    a.add_argument('--dims_mlp', type=int, nargs='+', default=None)
    a.add_argument('--time_embedding_size', type=int, default=None)
    a.add_argument('--n_steps', type=int, default=None)
    a.add_argument('--s_steps', type=int, default=None)
    a.add_argument('--mlp_dropout', type=float, default=None)
    a.add_argument('--act_func', type=str, default=None, choices=['tanh', 'relu', 'sigmoid'])
    a.add_argument('--norm', type=str2bool, nargs='?', const=True, default=None)

    r = p.add_argument_group('objective')
    r.add_argument('--rank_lambda', type=float, default=None)
    r.add_argument('--rank_num_neg', type=int, default=None)
    r.add_argument('--rank_tau', type=float, default=None)

    z = p.add_argument_group('source')
    z.add_argument('--alpha', type=float, default=None)
    z.add_argument('--kappa', type=int, default=None)

    o = p.add_argument_group('optimization')
    o.add_argument('--batch_size', type=int, default=None)
    o.add_argument('--eval_batch_size', type=int, default=None)
    o.add_argument('--max_epochs', type=int, default=None)
    o.add_argument('--patience', type=int, default=None)
    o.add_argument('--lr', type=float, default=None)
    o.add_argument('--weight_decay', type=float, default=None)

    m = p.add_argument_group('misc')
    m.add_argument('--seed', type=int, default=None)
    m.add_argument('--eval_every', type=int, default=None)
    m.add_argument('--gpu', type=int, default=None)
    m.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=None)
    m.add_argument('--save_path', type=str, default=None)
    m.add_argument('--log_path', type=str, default=None)
    return p


def apply_cli_overrides(cfg, ns):
    for k, v in vars(ns).items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise ValueError(f"[cli] unknown CONFIG key '{k}'")
        setattr(cfg, k, v)
    return cfg


def main():
    ns = build_parser().parse_args()
    args = apply_cli_overrides(CONFIG, ns)

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    sys.stdout = FileLogger(args.log_path)
    print(f"Config: {vars(args)}")

    device = resolve_device(int(args.gpu))
    set_seed(args.seed, deterministic=args.deterministic)

    train_df = load_interactions(args.train_path)
    valid_df = load_interactions(args.valid_path)
    test_df = load_interactions(args.test_path)
    user_map, item_map = build_mappings_from_train(train_df)
    num_users, num_items = len(user_map), len(item_map)

    train_csr = df_to_csr(train_df, user_map, item_map, num_users, num_items)
    valid_csr = df_to_csr(valid_df, user_map, item_map, num_users, num_items)
    test_csr = df_to_csr(test_df, user_map, item_map, num_users, num_items)
    model = ReFlow(num_users, num_items, train_csr, args, device).to(device)

    train_reflow(args, model, train_csr, valid_csr, test_csr, num_users, device)


if __name__ == '__main__':
    main()
